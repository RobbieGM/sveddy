#!/usr/bin/env bash

# pg_config on NixOS returns some wrong outputs
# this wrapper script aims to correct that

set -euo pipefail

INCLUDEDIR=$(nix-store -r $(which postgres) 2>/dev/null)/include

if [[ $# -eq 0 ]]; then
    pg_config | grep -v INCLUDEDIR
    echo "INCLUDEDIR = $INCLUDEDIR"
    echo "PKGINCLUDEDIR = $INCLUDEDIR"
    echo "INCLUDEDIR-SERVER = $INCLUDEDIR/server"
    exit 0
fi

# Handle specific arguments
for arg in "$@"
do
    case $arg in
        --includedir) echo $INCLUDEDIR; break;;
        --pkgincludedir) echo $INCLUDEDIR; break;;
        --includedir-server) echo "$INCLUDEDIR/server"; break;;
        *) pg_config $arg; break;;
    esac
done
