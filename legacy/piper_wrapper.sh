#!/usr/bin/env bash
DIR_LIB="/usr/local/share/piper"
export LD_LIBRARY_PATH="${DIR_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec /usr/local/bin/piper "$@"
