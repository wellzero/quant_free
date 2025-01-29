#!/bin/bash

# Set the model and API key based on the provided option
if [ "$1" == "deepseek" ]; then
    aider --model deepseek --watch-files --api-key "deepseek=sk-8a511a6d205b47f5b4343dbded938830"
elif [ "$1" == "qwen" ]; then
    aider --model openrouter/qwen/qwen-2.5-coder-32b-instruct --watch-files --api-key "openrouter=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"
else
    echo "Please specify a model: deepseek or qwen"
    exit 1
fi
