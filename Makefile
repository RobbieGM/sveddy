MODULES = sveddy
EXTENSION = sveddy
DATA = sveddy--0.1.0.sql
REGRESS = uv_init uv_insert_trigger uv_weight_initialization uv_garbage_collection uv_predict uv_train
REGRESS_OPTS = --port=5433 --host=localhost
PG_CFLAGS =
ifneq ($(SSE),no)
   PG_CFLAGS += -msse3
endif
# Check if nix exists, and set PG_CONFIG accordingly
IS_NIX := $(shell which nix > /dev/null 2>&1 && echo yes || echo no)
ifeq ($(IS_NIX),yes)
   PG_CONFIG = ./pg_config_nix
else
   PG_CONFIG = pg_config
endif
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
