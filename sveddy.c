#include <math.h>
#include "postgres.h"
#include <float.h>
#include "executor/spi.h"
#include "fmgr.h"
#include "utils/array.h"
#include "utils/lsyscache.h"
#include <pthread.h>
#include <omp.h>
#include <string.h>
#if defined(__i386__) || defined(__x86_64__)
#include "immintrin.h"
#endif
// Ignore any errors coming from the <omp.h> file if using clangd language server with gcc's omp.h

#define ROWS_TO_FETCH 2048


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
static float4 dot(const float4 *a, const float4 *b, int k) {
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

/**
 * Solves a linear equation of the form Ax = b where A is PSD and symmetric.
 * A is n by n and b is n by 1. Data is returned in x.
*/
static void cholesky(float4 *A, float4 *b, float4 *x, int n) {
	int i,j,k;
	float4 *L = alloca(n * n * sizeof(float4));
	float4 *y;

	// Cholesky decomposition
	for (i = 0; i < n; i++) {
		for (j = 0; j < (i+1); j++) {
			float4 sum = 0.0;
			for (k = 0; k < j; k++)
				sum += L[i * n + k] * L[j * n + k];
			L[i * n + j] = (i == j) ? 
						sqrt(A[i * n + i] - sum) : 
						(1.0 / L[j * n + j] * (A[i * n + j] - sum));
		}
	}

	// Solve L*y = b
	y = alloca(n * sizeof(float4));
	for (i = 0; i < n; i++) {
		float4 sum = 0;
		for (k = 0; k < i; k++)
			sum += L[i * n + k] * y[k];
		y[i] = (b[i] - sum) / L[i * n + i];
	}

	// Solve L^T*x = y
	for (i = n - 1; i >= 0; i--) {
		float4 sum = 0;
		for (k = i + 1; k < n; k++)
			sum += L[k * n + i] * x[k];
		x[i] = (y[i] - sum) / L[i * n + i];
	}
}


PG_FUNCTION_INFO_V1(update_model_uv);
Datum update_model_uv(PG_FUNCTION_ARGS) {
	TriggerData *trigger_data = (TriggerData *) fcinfo->context;
	HeapTuple return_tuple = TRIGGER_FIRED_BY_UPDATE(trigger_data->tg_event) ? trigger_data->tg_newtuple : trigger_data->tg_trigtuple;
	char *source_table = SPI_getrelname(trigger_data->tg_relation);
	char sql[1024];
	char *user_column, *item_column, *rating_column, *u_table, *v_table;
	int user_column_number_in_source_table;
	int item_column_number_in_source_table;
	int32 user_id, item_id;
	int32 k;
	float4 regularization_factor;
	int ret;
	bool is_null;
	if (!CALLED_AS_TRIGGER(fcinfo)) {
		elog(ERROR, "update_model_uv: not called as trigger");
	}

	// Find user_column, item_column, model_table
	sprintf(sql, "SELECT user_column, item_column, rating_column, u_table, v_table, k, regularization_factor FROM sveddy_models_uv WHERE source_table = '%s'", source_table);
	SPI_connect();
	ret = SPI_exec(sql, 1);
	if (ret != SPI_OK_SELECT) {
		elog(ERROR, "Could not find source_table = '%s' in sveddy_models_uv", source_table);
	}
	user_column = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1);
	item_column = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2);
	rating_column = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 3);
	u_table = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 4);
	v_table = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 5);
	k = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 6, &is_null));
	regularization_factor = DatumGetFloat4(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 7, &is_null));
	if (is_null) {
		elog(ERROR, "update_model_uv: could not retrieve k");
	}
	
	// Find user_id, item_id in the triggering row
	user_column_number_in_source_table = SPI_fnumber(trigger_data->tg_relation->rd_att, user_column);
	if (user_column_number_in_source_table == SPI_ERROR_NOATTRIBUTE) {
		elog(ERROR, "update_model_uv: could not find column %s in table %s", user_column, source_table);
	}
	item_column_number_in_source_table = SPI_fnumber(trigger_data->tg_relation->rd_att, item_column);
	if (item_column_number_in_source_table == SPI_ERROR_NOATTRIBUTE) {
		elog(ERROR, "update_model_uv: could not find column %s in table %s", item_column, source_table);
	}
	user_id = DatumGetInt32(SPI_getbinval(return_tuple, trigger_data->tg_relation->rd_att, user_column_number_in_source_table, &is_null));
	if (is_null) {
		elog(ERROR, "update_model_uv: user id in inserted row is null");
	}
	item_id = DatumGetInt32(SPI_getbinval(return_tuple, trigger_data->tg_relation->rd_att, item_column_number_in_source_table, &is_null));
	if (is_null) {
		elog(ERROR, "update_model_uv: item id in inserted row is null");
	}


	// Add row to U or V if necessary
	if (TRIGGER_FIRED_BY_INSERT(trigger_data->tg_event)) {
		sprintf(sql, "INSERT INTO %s (id, weights) VALUES (%d, get_initial_weights_uv(%d)) ON CONFLICT (id) DO NOTHING", u_table, user_id, k);
		ret = SPI_exec(sql, 0);
		if (ret != SPI_OK_INSERT) {
			elog(ERROR, "Failed to insert new row into %s", u_table);
		}
		sprintf(sql, "INSERT INTO %s (id, weights) VALUES (%d, get_initial_weights_uv(%d)) ON CONFLICT (id) DO NOTHING", v_table, item_id, k);
		ret = SPI_exec(sql, 0);
		if (ret != SPI_OK_INSERT) {
			elog(ERROR, "Failed to insert new row into %s", v_table);
		}
	}

	// Recalculate the entries in u_table and v_table corresponding to the user id and item id
	for (enum uv_table_which which = U; which <= V; which++) {
		char *this_table, *that_table;
		int32 this_id;
		char *this_id_column, *that_id_column;
		SPIParseOpenOptions parse_open_options = {0};
		Portal cursor;
		float *A, *b;
		float *weights;
		if (which == U) {
			this_table = u_table;
			that_table = v_table;
			this_id = user_id;
			this_id_column = user_column;
			that_id_column = item_column;
		} else {
			this_table = v_table;
			that_table = u_table;
			this_id = item_id;
			this_id_column = item_column;
			that_id_column = user_column;
		}
		// Initialize A, b
		A = palloc(sizeof(float4) * k * k);
		b = palloc(sizeof(float4) * k);
		weights = palloc(sizeof(float4) * k);
		memset(A, 0, sizeof(float4) * k * k);
		memset(b, 0, sizeof(float4) * k);
		// Fill in the diagonal elements with regularization_factor
		// diagonal elements are k+1 indices apart
		for (float4 *element = A; element < A + k * k; element += k + 1) {
			*element = regularization_factor;
		}
		// Calculate new weights
		sprintf(sql, "SELECT \"%s\"::real, \"%s\".weights FROM \"%s\""
		  " JOIN \"%s\" ON \"%s\".id = \"%s\".\"%s\""
		  " WHERE \"%s\".\"%s\" = %d",
		  rating_column, that_table, source_table,
		  that_table, that_table, source_table, that_id_column,
		  source_table, this_id_column, this_id
		);
		cursor = SPI_cursor_parse_open(NULL, sql, &parse_open_options);
		while (true) {
			// Get row(s) from the above query
			float4 rating;
			float4 *that_weights;
			ArrayType *that_weight_arr;
			SPI_cursor_fetch(cursor, true, ROWS_TO_FETCH); // fetching a higher number of rows at a time seems to increase perf
			// TODO this might be parallelizable to some degree
			for (int i = 0; i < SPI_processed; i++) {
				rating = DatumGetFloat4(SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1, &is_null));
				that_weight_arr = DatumGetArrayTypeP(SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 2, &is_null));
				that_weights = (float4 *) ARR_DATA_PTR(that_weight_arr);
				for (int j = 0; j < k; j++) {
					if (isnan(that_weights[i]))
						elog(WARNING, "i = %d, rating = %f", i, rating);
				}
				// Add that_weights * that_weights^T to A (TODO: SIMD)
				for (int row = 0; row < k; row++) {
					for (int col = 0; col < k; col++) {
						size_t offset = row * k + col;
						A[offset] += that_weights[row] * that_weights[col];
					}
				}

				// Add rating * that_weights to the right b vector
				b = A + k * k;
				for (int row = 0; row < k; row++) {
					size_t offset = row;
					b[offset] += rating * that_weights[row];
				}
			}
			if (SPI_processed == 0) break;
			SPI_freetuptable(SPI_tuptable);
		}
		
		// Solve A(weights) = b
		cholesky(A, b, weights, k);

		// Write results
		{
			Datum *weights_datums = palloc(sizeof(Datum) * k);
			ArrayType *weights_array;
			Datum datums[2];
			Oid argtypes[2] = {FLOAT4ARRAYOID, INT4OID};
			for (int j = 0; j < k; j++) {
				weights_datums[j] = Float4GetDatum(weights[j]);
			}
			weights_array = construct_array(weights_datums, k, FLOAT4OID, sizeof(float4), true, 'i');
			datums[0] = PointerGetDatum(weights_array);
			datums[1] = Int32GetDatum(this_id);
			sprintf(sql, "UPDATE %s SET weights = $1 WHERE id = $2", this_table);
			ret = SPI_execute_with_args(sql, 2, argtypes, datums, NULL, false, 0);
			if (ret != SPI_OK_UPDATE) {
				elog(ERROR, "trigger update_model_uv: failed row update for %c table, id = %d, ret = %d", which == U ? 'u' : 'v', this_id, ret);
			}
		}
	}
	
	SPI_finish();
	PG_RETURN_POINTER(return_tuple);
}

static bool is_validation_row(int32 user_id, int32 item_id, float validation_fraction) {
	// LCG Park-Miller
	unsigned int seed = user_id << 16 | item_id;
	seed = (seed * 48271) % 0x7fffffff;
	return (seed / (float) 0x7fffffff) < validation_fraction;
}

PG_FUNCTION_INFO_V1(train_uv);
Datum train_uv(PG_FUNCTION_ARGS) {
	// TODO handle case where not all of the data fits into memory
	char *source_table = PG_GETARG_CSTRING(0);
	int32 patience = PG_GETARG_INT16(1);
	int32 max_iterations = PG_GETARG_INT16(2);
	float4 validation_fraction = PG_GETARG_FLOAT4(3);
	bool quiet = PG_GETARG_BOOL(4);
	char *user_column, *item_column, *rating_column, *u_table, *v_table;
	float4 regularization_factor;
	int32 k;
	char sql[800];
	int ret;
	uint32 max_id_uv, max_id_u, max_id_v;
	float4 *linear_equations;
	size_t linear_equations_size;
	float4 *u, *v;
	Portal cursor;
	int current_patience = patience;
	double best_rmse = DBL_MAX;
	SPIParseOpenOptions parse_open_options = {0};
	SPIPlanPtr update_u_plan, update_v_plan;
	Oid update_uv_plan_argument_oids[2] = {FLOAT4ARRAYOID, INT4OID};
	bool is_null;

	SPI_connect();
	// Load model parameters
	sprintf(sql, "SELECT user_column, item_column, rating_column, u_table, v_table, k, regularization_factor FROM sveddy_models_uv WHERE source_table = '%s'", source_table);
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
	regularization_factor = DatumGetFloat4(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 7, &is_null));
	// Prepare query plans for statements to update rows in u, v
	sprintf(sql, "UPDATE %s SET weights = $1 WHERE id = $2", u_table);
	update_u_plan = SPI_prepare(sql, 2, update_uv_plan_argument_oids);
	sprintf(sql, "UPDATE %s SET weights = $1 WHERE id = $2", v_table);
	update_v_plan = SPI_prepare(sql, 2, update_uv_plan_argument_oids);

	// Find max(user id), max(item id) to figure out how much memory to allocate
	// This assumes that all of the linear equations will fit into memory at once,
	// which is true for up to 4.6 million users, k = 20 with 8GB of RAM.
	// (not counting the memory used by u and v)
	sprintf(sql, "SELECT max(id) FROM \"%s\"", u_table);
	ret = SPI_exec(sql, 1);
	if (ret != SPI_OK_SELECT) {
		elog(ERROR, "train_uv: Failed query: %s", sql);
	}
	max_id_u = DatumGetUInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &is_null));
	sprintf(sql, "SELECT max(id) FROM \"%s\"", v_table);
	ret = SPI_exec(sql, 1);
	if (ret != SPI_OK_SELECT) {
		elog(ERROR, "train_uv: Failed query: %s", sql);
	}
	max_id_v = DatumGetUInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &is_null));
	max_id_uv = max_id_u > max_id_v ? max_id_u : max_id_v;

	// Allocate enough memory for max_rows_uv linear equations
	// each linear equation takes sizeof(float4) * k * k (for A) + sizeof(float4) * k (for B) bytes
	// the related As and bs will be stored contiguously, with the matrix A before b
	// |-----------------|---------------|-----------------|-------------- ...
	//   A0 (k*k*float4)   b0 (k*float4)   A1 (k*k*float4)   b1 (k*float4)  ...
	// The index in linear_equations is equivalent to either the user id (if updating U) or the item id (if updating V)
	linear_equations_size = sizeof(float4) * (max_id_uv + 1) * k * (k+1);
	linear_equations = palloc(linear_equations_size);

	// Allocate memory for U and V
	u = palloc(sizeof(float4) * (max_id_u + 1) * k);
	v = palloc(sizeof(float4) * (max_id_v + 1) * k);
	// Fill with 0xff (NaN) to indicate that the entry is not present, as some rows may have been deleted
	memset(u, 0xff, sizeof(float4) * (max_id_u + 1) * k);
	memset(v, 0xff, sizeof(float4) * (max_id_v + 1) * k);
	
	// Fill u and v
	for (enum uv_table_which which = U; which <= V; which++) {
		float4 *this_memblock = which == U ? u : v;
		sprintf(sql, "SELECT id, weights FROM \"%s\"", which == U ? u_table : v_table);
		cursor = SPI_cursor_parse_open(NULL, sql, &parse_open_options);
		while (true) {
			// Get row(s) from the above query
			int32 id;
			ArrayType *weights_arr;
			float4 *weights;
			SPI_cursor_fetch(cursor, true, ROWS_TO_FETCH);
			// TODO this might be parallelizable to some degree
			for (int i = 0; i < SPI_processed; i++) {
				id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1, &is_null));
				weights_arr = DatumGetArrayTypeP(SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 2, &is_null));
				weights = (float4 *) ARR_DATA_PTR(weights_arr);
				memcpy(this_memblock + id * k, weights, sizeof(float4) * k);
			}
			if (SPI_processed == 0) break;
			SPI_freetuptable(SPI_tuptable);
		}
	}

	// Do one pass through the training data for each step
	for (int training_iteration = 0; training_iteration < max_iterations && current_patience > 0; training_iteration++) {
		enum uv_table_which which = training_iteration % 2; // Start arbitrarily with the u step, alternating
		double total_square_error_train = 0;
		double total_square_error_validation = 0;
		double rmse_train, rmse_validation, rmse_used;
		int total_processed_train = 0;
		int total_processed_validation = 0;
		char *this_id_column = which == U ? user_column : item_column;
		char *that_id_column = which == U ? item_column : user_column;
		float4 *this_memblock = which == U ? u : v;
		float4 *that_memblock = which == U ? v : u;
		int this_max_id = which == U ? max_id_u : max_id_v;

		// This query returns ratings from source_table along with the
		// corresponding user and item ids
		sprintf(sql, "SELECT \"%s\".\"%s\"::real, \"%s\".\"%s\", \"%s\".\"%s\" FROM \"%s\"",
		  source_table, rating_column, source_table, this_id_column, source_table, that_id_column, source_table
		);

		// A is at the beginning of each [A, b] block in linear_equations
		// Clear A matrices, setting diagonal elements to regularization_factor
		memset(linear_equations, 0, linear_equations_size);
		#pragma omp parallel for
		for (float4 *A = linear_equations; A < linear_equations + (this_max_id + 1) * k * (k+1); A += k * (k+1)) {
			// Fill in the diagonal elements with regularization_factor
			// diagonal elements are k+1 indices apart
			for (float4 *element = A; element < A + k * k; element += k + 1) {
				*element = regularization_factor;
			}
		}

		// Now sum other components to A matrices and b vectors
		cursor = SPI_cursor_parse_open(NULL, sql, &parse_open_options);
		while (true) {
			// Get row(s) from the above query
			// about 46 seconds for 100 million rows
			SPI_cursor_fetch(cursor, true, ROWS_TO_FETCH);
			if (SPI_processed == 0) break;
			// also about 42 seconds for 100 million rows
			// trying to parallelize by using (this_max_id + 1) locks makes it slower (4 minutes)
			// using many threads for each chunk of loaded data, with each thread only handling this_id % num_threads == thread_num, is also worse (5 minutes)
			for (int i = 0; i < SPI_processed; i++) {
				float4 rating = DatumGetFloat4(SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1, &is_null));
				int32 this_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 2, &is_null));
				int32 that_id = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 3, &is_null));
				float4 *this_weights = this_memblock + this_id * k;
				float4 *that_weights = that_memblock + that_id * k;
				float4 *A, *b;
				float error = rating - dot(this_weights, that_weights, k);
				if (is_validation_row(this_id, that_id, validation_fraction)) {
					total_square_error_validation += error * error;
					total_processed_validation++;
				} else {
					// Add that_weights * that_weights^T to the right A matrix (TODO: SIMD)
					A = linear_equations + k * (k + 1) * this_id;
					for (int row = 0; row < k; row++) {
						for (int col = 0; col < k; col++) {
							size_t offset = row * k + col;
							A[offset] += that_weights[row] * that_weights[col];
						}
					}

					// Add rating * that_weights to the right b vector
					b = A + k * k;
					for (int row = 0; row < k; row++) {
						size_t offset = row;
						b[offset] += rating * that_weights[row];
					}

					total_square_error_train += error * error;
					total_processed_train++;
				}
			}
			SPI_freetuptable(SPI_tuptable);
		}

		// Update patience and report RMSE
		rmse_train = sqrt(total_square_error_train / total_processed_train);
		rmse_validation = sqrt(total_square_error_validation / total_processed_validation);
		rmse_used = validation_fraction == 0 ? rmse_train : rmse_validation;
		if (rmse_used > best_rmse) {
			current_patience--;
		} else {
			current_patience = patience;
			best_rmse = rmse_used;
		}
		if (!quiet) {
			if (validation_fraction == 0) {
				elog(INFO, "train_uv: RMSE (train) = %f (for %c step)", rmse_train, which == U ? 'u' : 'v');
			} else {
				elog(INFO, "train_uv: RMSE (train, validation) = %f, %f (for %c step)", rmse_train, rmse_validation, which == U ? 'u' : 'v');
			}
		}

		// Solve all linear equations, placing solutions into u or v
		#pragma omp parallel for
		for (int id = 0; id <= this_max_id; id++) {
			float4 *weights = this_memblock + id * k;
			float4 *A = linear_equations + (k * k + k) * id;
			float4 *b = A + k * k;
			if (isnanf(weights[0])) {
				// This row has been deleted
				continue;
			}
			// Compute weights
			cholesky(A, b, weights, k); // This function could be refactored to write directly into b, avoiding memcpy
		}
	}

	// Write results
	for (enum uv_table_which which = U; which <= V; which++) {
		SPIPlanPtr update_plan = which == U ? update_u_plan : update_v_plan;
		int this_max_id = which == U ? max_id_u : max_id_v;
		float4 *this_memblock = which == U ? u : v;
		for (int id = 0; id <= this_max_id; id++) {
			float4 *weights = this_memblock + id * k;
			Datum *weights_datums = palloc(sizeof(Datum) * k);
			ArrayType *weights_array;
			Datum datums[2];
			if (isnanf(weights[0])) {
				// This row has been deleted
				continue;
			}
			// Create array of datums from weights
			for (int j = 0; j < k; j++) {
				weights_datums[j] = Float4GetDatum(weights[j]);
			}
			weights_array = construct_array(weights_datums, k, FLOAT4OID, sizeof(float4), true, 'i');
			datums[0] = PointerGetDatum(weights_array);
			datums[1] = Int32GetDatum(id);
			// Update row
			ret = SPI_execute_plan(update_plan, datums, NULL, false, 0);
			if (ret != SPI_OK_UPDATE) {
				elog(ERROR, "train_uv: failed row update for %c table, id = %d, ret = %d", which == U ? 'u' : 'v', id, ret);
			}
			pfree(weights_datums);
		}
	}
	pfree(linear_equations);
	pfree(u);
	pfree(v);
	SPI_finish();
	PG_RETURN_NULL();
}

