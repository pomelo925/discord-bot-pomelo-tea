#!/bin/bash

export DISPLAY=:0

cd docker
docker compose -p discord-bot up default -d
# docker compose -p discord-bot up devel -d