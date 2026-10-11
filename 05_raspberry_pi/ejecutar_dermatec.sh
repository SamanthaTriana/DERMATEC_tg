#!/usr/bin/env bash
# Abre la app DERMATEC V6 con el entorno de la Raspberry
cd "$(dirname "$0")/.."
.venv/bin/python dermatec_app_v6.py "$@" 2>&1 | tee -a "$HOME/dermatec_ultima_ejecucion.log"
