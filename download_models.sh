#!/bin/bash

mkdir -p /cache
mkdir -p /userdata

exec streamlit run app.py --server.fileWatcherType=none
