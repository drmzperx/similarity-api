#!/bin/sh
#conda run -n similarity-api fastapi dev main.py
conda run -n similarity-api uvicorn main:app --host 0.0.0.0 --port 5000 --workers 4