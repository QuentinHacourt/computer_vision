#!/usr/bin/env sh

# if -i || venv doesn't exist
DIR="./venv/"

if [ ! -d "$DIR" ] || [ "$1" = "-i" ]; then
    echo "Setting up virtual environment :)"
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

venv/bin/python code/main.py
