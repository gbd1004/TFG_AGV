#!/bin/bash

pylint --fail-under=7.0 --recursive=y -f text services/forecasting_cpu services/forecasting_gpu services/reciever services/simulator