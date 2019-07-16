#!/bin/bash

CASE='upwelling'

echo ' '
echo $CASE
echo ' '

infile=../$CASE.log

grep E+ $infile | grep : | grep E- | sed "s/:\|-/ /g" | sed "s/E /E-/g" > ../output/$CASE``_ek.out
# sed troca todos os ':' e '-' por ' ' 

# grep E+ $infile | grep : | grep E- | sed "s/:/ /g"  > ../output/$CASE``_ek.out

# sed "1,329 d" $infile | grep -v WRT | grep -v 'GET_2DFLD' | grep -v 'File' | grep -v 'Tmin' | grep -v 'Min =' | sed "s/:/ /g" > ek.out
