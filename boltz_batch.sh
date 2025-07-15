#!/bin/bash

# Utility script to fire off sequential Boltz-2 predictions

for i in *.yaml;
    do
        prefix=$(basename "$i" .yaml)
        boltz predict "$i" > "${prefix}.log"
done

echo "All calculations completed comrade"