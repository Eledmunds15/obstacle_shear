#!/bin/bash

# Shell script to transfer sync 000_data with HPC
USER="mtp24ele"
HOST="stanage.shef.ac.uk"
SOURCE_DIR="/home/Ethan/Projects/atom_sims/precipitate_calculations/"
DEST_DIR="/mnt/parscratch/users/mtp24ele/precipitate_calculations/"

# Use rsync for efficient transfer
echo "Starting file transfer..."
rsync -avz --exclude=".git" --exclude="000_archived_data" "$SOURCE_DIR/" "$USER@$HOST:$DEST_DIR"

# Check if the transfer was successful
if [ $? -eq 0 ]; then
  echo "Transfer completed successfully! 🎉"
else
  echo "Transfer failed. 😔 Please check your connection and paths."
  exit 1
fi
