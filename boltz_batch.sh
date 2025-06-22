#!/bin/bash

# Utility script to fire off sequential Boltz-2 predictions

for i in *.yaml;
    do
        boltz predict $i > {$i}.log
done

echo "All calculations completed comrade"