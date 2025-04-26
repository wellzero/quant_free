#!/bin/bash


# Set the model and API key based on the provided option
if [ "$1" == "ds" ]; then
    aider --model deepseek --watch-files --api-key "deepseek=$DEEPSEEK_API_KEY"
elif [ "$1" == "orr1" ]; then
    aider --model openrouter/deepseek/deepseek-r1 --watch-files --api-key "openrouter=$OPENROUTER_API_KEY"
elif [ "$1" == "orv3" ]; then
    aider --model openrouter/deepseek/deepseek-chat --watch-files --api-key "openrouter=$OPENROUTER_API_KEY"
elif [ "$1" == "qwq-32b" ]; then
    aider --model openai/qwq-32b --watch-files
elif [ "$1" == "qw" ]; then
    aider --model openrouter/qwen/qwen-2.5-coder-32b-instruct --watch-files --api-key "openrouter=$OPENROUTER_API_KEY" 
elif [ "$1" == "claude" ]; then
    #aider --model openrouter/anthropic/claude-3.5-sonnet --watch-files --api-key "openrouter=$OPENROUTER_API_KEY" 
    aider --model openrouter/anthropic/claude-3.7-sonnet --watch-files --api-key "openrouter=$OPENROUTER_API_KEY" 
else
    echo "Please specify a model: ds orr1 orv1 qw qwq-32b claude"
    exit 1
fi
