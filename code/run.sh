#!/usr/bin/env sh

DIR="venv"

if [ ! -d "$DIR" ] || [ "$1" == "-i" ]; then
    echo "Setting up Virtual environment :)"

    [ -d "$DIR" ] && rm -rf "$DIR"

    python3 -m venv venv
    . venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Using existing environment :)"
    . venv/bin/activate
fi

venv/bin/python3 code/main.py
