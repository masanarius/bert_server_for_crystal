#!/bin/bash
uvicorn bert_server:app --host 0.0.0.0 --port ${PORT:-10000}
