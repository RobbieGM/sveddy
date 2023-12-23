#include "postgres.h"
#include "executor/spi.h"
#include "fmgr.h"
#include "utils/array.h"
#include "utils/lsyscache.h"

// void _PG_init()
// https://www.postgresql.org/docs/12/trigger-example.html
// https://www.postgresql.org/docs/current/trigger-interface.html

PG_MODULE_MAGIC;

static float4 random_uniform() {
	return (rand() / (float4) RAND_MAX);
}

PG_FUNCTION_INFO_V1(get_initial_weights);
Datum get_initial_weights(PG_FUNCTION_ARGS) {
	int32 k = PG_GETARG_INT32(0);
	// Return an array of length k filled with zeros.
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

enum uv_table_which {U, V};

static Datum update_u_or_v(PG_FUNCTION_ARGS, enum uv_table_which which) {
	TriggerData *trigger_data = (TriggerData *) fcinfo->context;
	HeapTuple return_tuple = TRIGGER_FIRED_BY_UPDATE(trigger_data->tg_event) ? trigger_data->tg_newtuple : trigger_data->tg_trigtuple;
	char *source_table_name = SPI_getrelname(trigger_data->tg_relation);
	char sql[256];
	char *user_column, *item_column, *rating_column, *u_table, *v_table;
	char *source_id_column;
	char *model_table;
	char which_char = which == U ? 'u' : 'v';
	int32 k;
	int ret;
	bool is_null;
	if (!CALLED_AS_TRIGGER(fcinfo)) {
		elog(ERROR, "update_%c: not called as trigger", which_char);
	}

	// Find user_column, item_column, rating_column, model_table
	sprintf(sql, "SELECT user_column, item_column, rating_column, u_table, v_table, k FROM sveddy_models_uv WHERE source_table = '%s'", source_table_name);
	SPI_connect();
	ret = SPI_exec(sql, 1);
	if (ret != SPI_OK_SELECT) {
		elog(ERROR, "Could not find source_table = '%s' in sveddy_models_uv", source_table_name);
	}
	user_column = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1);
	item_column = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2);
	rating_column = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 3);
	u_table = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 4);
	v_table = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 5);
	k = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 6, &is_null));
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
		sprintf(sql, "INSERT INTO %s (id, weights) VALUES (%d, get_initial_weights(%d)) ON CONFLICT (id) DO NOTHING", model_table, id, k);
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
