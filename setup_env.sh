#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VENV_DIR="$PROJECT_ROOT/venv"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"
ACTIVATE_SCRIPT=""
VENV_PYTHON=""

if [ -d "$VENV_DIR" ]; then
  echo "Ambiente virtual '$VENV_DIR' ja existe. Reutilizando..."
else
  echo "Criando ambiente virtual em '$VENV_DIR'..."
  python3 -m venv "$VENV_DIR"
fi

echo "Criando ambiente virtual em '$VENV_DIR'..."
python3 -m venv "$VENV_DIR"

echo "Ativando ambiente virtual..."
if [ -f "$VENV_DIR/bin/activate" ]; then
  ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
  VENV_PYTHON="$VENV_DIR/bin/python"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
  ACTIVATE_SCRIPT="$VENV_DIR/Scripts/activate"
  VENV_PYTHON="$VENV_DIR/Scripts/python"
else
  echo "Nao foi possivel encontrar o script de ativacao da virtualenv."
  exit 1
fi

# shellcheck disable=SC1091
source "$ACTIVATE_SCRIPT"

echo "Atualizando pip..."
"$VENV_PYTHON" -m pip install --upgrade pip

echo "Instalando dependencias de requirements.txt..."
"$VENV_PYTHON" -m pip install -r "$REQUIREMENTS_FILE"

if "$VENV_PYTHON" -m pip show ipykernel &>/dev/null; then
  echo "Registrando kernel no Jupyter..."
  "$VENV_PYTHON" -m ipykernel install --user --name=venv --display-name "Python ($(basename $PROJECT_ROOT))"
else
  echo "ipykernel nao encontrado, pulando registro do kernel."
fi


echo "Concluido."

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Aviso: como o script foi executado com 'bash', a ativacao nao persiste no shell atual."
  echo "Para manter a venv ativa no seu terminal, rode: source utils/setup_env.sh"
fi
