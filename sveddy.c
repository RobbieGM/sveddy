#include <math.h>
#include "postgres.h"
#include "executor/spi.h"
#include "fmgr.h"
#include "utils/array.h"
#include "utils/lsyscache.h"

#if defined(__i386__) || defined(__x86_64__)
#include "immintrin.h"
#endif

#define ROWS_TO_FETCH 128

// void _PG_init()
// https://www.postgresql.org/docs/12/trigger-example.html
// https://www.postgresql.org/docs/current/trigger-interface.html

PG_MODULE_MAGIC;

static float4 random_uniform() {
	return (rand() / (float4) RAND_MAX);
}

PG_FUNCTION_INFO_V1(get_initial_weights_uv);
Datum get_initial_weights_uv(PG_FUNCTION_ARGS) {
	int32 k = PG_GETARG_INT32(0);
	// Return an array of length k filled with random numbers.
	ArrayType *result;
	int16 element_type_width;
	bool element_type_by_value;
	char element_type_alignment_code;
	Datum *values = palloc(sizeof(Datum) * k);
	for (int i = 0; i < k; i++) {
		float4 val = (float4) random_uniform();
		values[i] = Float4GetDatum(val);
	}
	get_typlenbyvalalign(FLOAT4OID, &element_type_width, &element_type_by_value, &element_type_alignment_code);
	result = construct_array(values, k, FLOAT4OID, element_type_width, element_type_by_value, element_type_alignment_code);
	PG_RETURN_ARRAYTYPE_P(result);
}


/*
 * Takes dot product of a and b, each of which are k-length.
 */
static float4 dot(float4 *a, float4 *b, int k) {
#if !defined(__i386__) && !defined(__x86_64__)
	float4 sum = 0;
	for (int i = 0; i < k; i++) {
		sum += a[i] * b[i];
	}
	return sum;
#else
	float4 sum = 0;
	__m128 sum_vec = _mm_setzero_ps(); // x is four floats, initialized to zero
	__m128 shuf_vec;
	int i;
	int simd_len = (k / 4) * 4; // Maximum length, chunked to four elements at a time, where each chunk is full
	for (i = 0; i < simd_len; i += 4) {
		__m128 vec1 = _mm_loadu_ps(a + i);
		__m128 vec2 = _mm_loadu_ps(b + i);
		__m128 multiplied = _mm_mul_ps(vec1, vec2);
		sum_vec = _mm_add_ps(sum_vec, multiplied);
	}
	shuf_vec = _mm_movehdup_ps(sum_vec);
	sum_vec = _mm_add_ps(sum_vec, shuf_vec);
	shuf_vec = _mm_movehl_ps(shuf_vec, sum_vec);
	sum_vec = _mm_add_ss(sum_vec, shuf_vec);
	sum = _mm_cvtss_f32(sum_vec);
	// Do the rest of the elements in case k wasn't divisible by four
	for (; i < k; i++) {
		sum += a[i] * b[i];
	}
	return sum;
#endif
}


PG_FUNCTION_INFO_V1(predict_uv);
Datum predict_uv(PG_FUNCTION_ARGS) {
	// The function takes two arguments, both of which are real[].
	// Returns the dot product, as a real.
	ArrayType *a = PG_GETARG_ARRAYTYPE_P(0);
	ArrayType *b = PG_GETARG_ARRAYTYPE_P(1);
	int a_len = ArrayGetNItems(ARR_NDIM(a), ARR_DIMS(a));
	int b_len = ArrayGetNItems(ARR_NDIM(b), ARR_DIMS(b));
	float4 *a_data, *b_data;
	float4 result;
	if (a_len != b_len) {
		elog(ERROR, "Both arrays must have the same number of elements");
	}
	a_data = (float4 *) ARR_DATA_PTR(a);
	b_data = (float4 *) ARR_DATA_PTR(b);
	result = dot(a_data, b_data, a_len);
	PG_RETURN_FLOAT4(result);
}


enum uv_table_which {U, V};

static Datum update_u_or_v(PG_FUNCTION_ARGS, enum uv_table_which which) {
	TriggerData *trigger_data = (TriggerData *) fcinfo->context;
	HeapTuple return_tuple = TRIGGER_FIRED_BY_UPDATE(trigger_data->tg_event) ? trigger_data->tg_newtuple : trigger_data->tg_trigtuple;
	char *source_table = SPI_getrelname(trigger_data->tg_relation);
	char sql[256];
	char *user_column, *item_column, *u_table, *v_table;
	char *source_id_column;
	char *model_table;
	char which_char = which == U ? 'u' : 'v';
	int32 k;
	int ret;
	bool is_null;
	if (!CALLED_AS_TRIGGER(fcinfo)) {
		elog(ERROR, "update_%c: not called as trigger", which_char);
	}

	// Find user_column, item_column, model_table
	sprintf(sql, "SELECT user_column, item_column, u_table, v_table, k FROM sveddy_models_uv WHERE source_table = '%s'", source_table);
	SPI_connect();
	ret = SPI_exec(sql, 1);
	if (ret != SPI_OK_SELECT) {
		elog(ERROR, "Could not find source_table = '%s' in sveddy_models_uv", source_table);
	}
	user_column = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1);
	item_column = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2);
	u_table = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 3);
	v_table = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 4);
	k = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 5, &is_null));
	if (is_null) {
		elog(ERROR, "update_%c: could not retrieve k", which_char);
	}
	model_table = which == U ? u_table : v_table;
	source_id_column = which == U ? user_column : item_column;

	if (TRIGGER_FIRED_BY_INSERT(trigger_data->tg_event)) {
		// Add row to U or V if necessary
		int source_id_column_number = SPI_fnumber(trigger_data->tg_relation->rd_att, source_id_column);
		int32 id = DatumGetInt32(SPI_getbinval(return_tuple, trigger_data->tg_relation->rd_att, source_id_column_number, &is_null));
		if (source_id_column_number == SPI_ERROR_NOATTRIBUTE) {
			elog(ERROR, "update_%c: could not find column %s in table %s", which_char, source_id_column, model_table);
		}
		if (is_null) {
			elog(ERROR, "update_%c: could not retrieve id", which_char);
		}
		sprintf(sql, "INSERT INTO %s (id, weights) VALUES (%d, get_initial_weights_uv(%d)) ON CONFLICT (id) DO NOTHING", model_table, id, k);
		ret = SPI_exec(sql, 0);
		if (ret != SPI_OK_INSERT) {
			elog(ERROR, "Failed to insert new row into %s", model_table);
		}
	}
	SPI_finish();
	PG_RETURN_POINTER(return_tuple);
}


PG_FUNCTION_INFO_V1(update_u);
Datum update_u(PG_FUNCTION_ARGS) {
	return update_u_or_v(fcinfo, U);
}

PG_FUNCTION_INFO_V1(update_v);
Datum update_v(PG_FUNCTION_ARGS) {
	return update_u_or_v(fcinfo, V);
}


/**
 * Solves a linear equation of the form Ax = b. A is k by k and b is k by 1.
 * Data is returned in x.
*/
static void gauss_jordan(float *A, float *b, float *x, int n) {
    // Gauss-Jordan elimination
    int i, j, k;
    float c, max;
    int idx;

    for (k = 0; k < n; k++) {
        // Pivoting
        max = 0.0;
        idx = k;
        for (i = k; i < n; i++) {
            if (max < fabs(A[i*n + k])) {
                max = fabs(A[i*n + k]);
                idx = i;
            }
        }

        // Exchanging rows
        if (idx != k) {
            for(j = 0; j < n; j++) {
                c = A[k*n + j];
                A[k*n + j] = A[idx*n + j];
                A[idx*n + j] = c;
            }

            c = b[k];
            b[k] = b[idx];
            b[idx] = c;
        }

        // Normalization
        c = A[k*n + k];
        for (j = k; j < n; j++) A[k*n + j] /= c;
        b[k] /= c;

        // Reduction
        for (i = 0; i < n; i++) {
            if (i != k) {
                c = A[i*n + k];
                for (j = k; j < n; j++) A[i*n + j] -= c * A[k*n + j];
                b[i] -= c * b[k];
            }
        }
    }
     
    for(i = 0; i < n; i++) {
        x[i] = b[i];
    }
}


PG_FUNCTION_INFO_V1(train_uv);
Datum train_uv(PG_FUNCTION_ARGS) {
	const int MAX_ITERATIONS = 8; // TODO dynamic number of iterations
	char *source_table = PG_GETARG_CSTRING(0);
	float4 regularization_factor = PG_GETARG_FLOAT4(1);
	bool quiet = PG_GETARG_BOOL(2);
	char *user_column, *item_column, *rating_column, *u_table, *v_table;
	int32 k;
	float4 learning_rate;
	char sql[800];
	int ret;
	uint32 max_rows_uv, rows_u, rows_v;
	uint32 max_id_uv, max_id_u, max_id_v;
	float4 *linear_equations; // TODO: all As are symmetric so lots of ram could be saved
	size_t linear_equations_size;
	uint32 *id_to_index_map, *index_to_id_map;
	uint32 index_to_id_map_size;
	SPIParseOpenOptions parse_open_options = {0};
	SPIPlanPtr update_u_plan, update_v_plan;
	Oid update_uv_plan_argument_oids[2] = {FLOAT4ARRAYOID, INT4OID};
	bool is_null;

	SPI_connect();
	sprintf(sql, "SELECT user_column, item_column, rating_column, u_table, v_table, k, learning_rate FROM sveddy_models_uv WHERE source_table = '%s'", source_table);
	ret = SPI_exec(sql, 1);
	if (ret != SPI_OK_SELECT) {
		elog(ERROR, "train_uv: Could not find source_table = '%s' in sveddy_models_uv", source_table);
	}
	user_column = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1);
	item_column = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2);
	rating_column = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 3);
	u_table = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 4);
	v_table = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 5);
	k = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 6, &is_null));
	learning_rate = 1; // DatumGetFloat4(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 6, &is_null)); // TODO
	// prepare query plans for statements to update rows in u, v
	sprintf(sql, "UPDATE %s SET weights = $1 WHERE id = $2", u_table);
	update_u_plan = SPI_prepare(sql, 2, update_uv_plan_argument_oids);
	sprintf(sql, "UPDATE %s SET weights = $1 WHERE id = $2", v_table);
	update_v_plan = SPI_prepare(sql, 2, update_uv_plan_argument_oids);

	// find max(# users, # items) to figure out how much memory to allocate
	// also find max user id and max item id for the id-to-index map
	// this assumes that all of the linear equations will fit into memory at once,
	// which is true for up to 4.6 million users, k = 20 with 8GB of RAM.
	sprintf(sql, "SELECT count(id), max(id) FROM \"%s\"", u_table);
	ret = SPI_exec(sql, 1);
	if (ret != SPI_OK_SELECT) {
		elog(ERROR, "train_uv: Failed query: %s", sql);
	}
	rows_u = DatumGetUInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &is_null));
	max_id_u = DatumGetUInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2, &is_null));

	sprintf(sql, "SELECT count(id), max(id) FROM \"%s\"", v_table);
	ret = SPI_exec(sql, 1);
	if (ret != SPI_OK_SELECT) {
		elog(ERROR, "train_uv: Failed query: %s", sql);
	}
	rows_v = DatumGetUInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &is_null));
	max_id_v = DatumGetUInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2, &is_null));

	max_rows_uv = rows_u > rows_v ? rows_u : rows_v;
	max_id_uv = max_id_u > max_id_v ? max_id_u : max_id_v;

	// allocate enough memory for max_rows_uv linear equations
	// each linear equation takes sizeof(float4) * k * k (for A) + sizeof(float4) * k (for B) bytes
	// the related As and bs will be stored contiguously, with the matrix A before b
	// |-----------------|---------------|-----------------|-------------- ...
	//   A0 (k*k*float4)   b0 (k*float4)   A1 (k*k*float4)   b1 (k*float4)  ...
	// The index in A is arbitrary and determined by which rows in the source
	// table are seen first. To figure out the index in A from a user id or
	// item id, the id_to_index_map and index_to_id_map arrays are created.
	linear_equations_size = sizeof(float4) * max_rows_uv * k * (k+1);
	linear_equations = palloc(linear_equations_size);
	// Allocate memory for index_to_id_map and id_to_index_map
	id_to_index_map = palloc(sizeof(uint32) * (max_id_uv + 1));
	index_to_id_map = palloc(sizeof(uint32) * max_rows_uv);

	// Do one pass through the training data for each step
	for (int training_iteration = 0; training_iteration < MAX_ITERATIONS; training_iteration++) {
		enum uv_table_which which = training_iteration % 2; // Start arbitrarily with the u step, alternating
		double total_square_error = 0;
		int total_processed = 0;
		char *this_table, *that_table;
		Portal cursor;
		if (which == U) {
			this_table = u_table;
			that_table = v_table;
		} else {
			this_table = v_table;
			that_table = u_table;
		}
		// This query returns ratings from source_table along with the
		// corresponding user and item weights, and user id (if u step)
		// or item id (if v step)
		sprintf(sql, "SELECT \"%s\".\"%s\"::real, \"%s\".weights, \"%s\".id::integer, \"%s\".weights FROM \"%s\""
		  " JOIN \"%s\" ON \"%s\".id = \"%s\".\"%s\""
		  " JOIN \"%s\" ON \"%s\".id = \"%s\".\"%s\""
		  " ORDER BY \"%s\".id",
		  source_table, rating_column, this_table, this_table, that_table, source_table,
		  u_table, u_table, source_table, user_column,
		  v_table, v_table, source_table, item_column,
		  this_table
		);
		// Set all values in the maps to -1 (actually 2**32-1) as a sentinel value
		// representing an empty map entry.
		index_to_id_map_size = 0;
		memset(id_to_index_map, -1, sizeof(uint32) * (max_id_uv + 1)); // representation of -1 is all 1 bits in the case of both char and int
		memset(index_to_id_map, -1, sizeof(uint32) * max_rows_uv);
		// Fill in linear_equations
		// Start by setting all As and bs to 0, then adding lambda*I to all A matrices
		memset(linear_equations, 0, linear_equations_size); // setting all bytes to zero also sets float4 numbers to zero
		// A is at the beginning of each [A, b] block in linear_equations
		// TODO this takes a long time (on the netflix dataset) and should be parallelized
		for (float4 *A = linear_equations; A < linear_equations + max_rows_uv * k * (k+1); A += k * (k+1)) {
			// Fill in the diagonal elements with regularization_factor
			// diagonal elements are k+1 indices apart
			for (float4 *element = A; element < A + k * k; element += k + 1) {
				*element = regularization_factor;
			}
		}
		// Now add other components to A matrices and b vectors
		cursor = SPI_cursor_parse_open(NULL, sql, &parse_open_options);
		while (true) {
			// Get row(s) from the above query
			float4 rating;
			float4 *this_weights;
			int32 id;
			float4 *that_weights;
			ArrayType *this_weight_arr, *that_weight_arr;
			size_t linear_equations_index;
			float4 *A, *b;
			SPI_cursor_fetch(cursor, true, ROWS_TO_FETCH); // fetching a higher number of rows at a time seems to increase perf
			for (int i = 0; i < SPI_processed; i++) {
				float4 error;
				rating = DatumGetFloat4(SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1, &is_null));
				this_weight_arr = DatumGetArrayTypeP(SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 2, &is_null));
				this_weights = (float4 *) ARR_DATA_PTR(this_weight_arr);
				id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 3, &is_null));
				that_weight_arr = DatumGetArrayTypeP(SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 4, &is_null));
				
				that_weights = (float4 *) ARR_DATA_PTR(that_weight_arr);
				// First find the index in linear_equations of the row by id
				if (id_to_index_map[id] != -1) {
					linear_equations_index = id_to_index_map[id];
				} else {
					// Create a new entry in both maps
					// The new index will one more than the current greatest
					linear_equations_index = index_to_id_map_size;
					index_to_id_map[index_to_id_map_size] = id;
					index_to_id_map_size++;
					id_to_index_map[id] = linear_equations_index;
				}
				// Add that_weights * that_weights^T to the right A matrix
				A = linear_equations + k * (k + 1) * linear_equations_index;
				for (int row = 0; row < k; row++) {
					for (int col = 0; col < k; col++) {
						size_t offset = row * k + col;
						A[offset] += that_weights[row] * that_weights[col];
					}
				}

				// Add rating * that_weights to the right b vector
				b = linear_equations + k * (k + 1) * linear_equations_index + k * k;
				for (int row = 0; row < k; row++) {
					size_t offset = row;
					b[offset] += rating * that_weights[row];
				}

				// Recalculate total_square_error
				error = rating - dot(this_weights, that_weights, k);
				total_square_error += error * error;
			}
			if (SPI_processed == 0) break;
			total_processed += SPI_processed;
			SPI_freetuptable(SPI_tuptable);
		}

		// Report RMSE
		if (!quiet) {
			elog(INFO, "train_uv: RMSE: %f (for %c step)", sqrt(total_square_error / total_processed), which == U ? 'u' : 'v');
		}

		// Solve all linear equations (TODO: parallelize)
		for (int i = 0; i < index_to_id_map_size; i++) {
			int id = index_to_id_map[i];
			float4 *A, *b;
			float4 *weights = palloc(sizeof(float4) * k);
			Datum *weights_datums = palloc(sizeof(Datum) * k);
			ArrayType *weights_array;
			Datum datums[2];
			SPIPlanPtr plan = which == U ? update_u_plan : update_v_plan;
			if (id == -1) {
				elog(ERROR, "train_uv: invalid index %d", i);
			}
			// compute weights
			A = linear_equations + (k * k + k) * i;
			b = A + k * k;
			gauss_jordan(A, b, weights, k);

			// Create array of datums from weights
			for (int j = 0; j < k; j++) {
				weights_datums[j] = Float4GetDatum(weights[j]);
			}
			weights_array = construct_array(weights_datums, k, FLOAT4OID, sizeof(float4), true, 'i');
			datums[0] = PointerGetDatum(weights_array);
			datums[1] = Int32GetDatum(id);
			// Update row
			ret = SPI_execute_plan(plan, datums, NULL, false, 0);
			if (ret != SPI_OK_UPDATE) {
				elog(ERROR, "train_uv: failed row update for %c table, id = %d, ret = %d", which == U ? 'u' : 'v', id, ret);
			}
		}

	}
	SPI_finish();
	pfree(linear_equations);
	pfree(id_to_index_map);
	pfree(index_to_id_map);
	PG_RETURN_NULL();
}
