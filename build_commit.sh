docker kill $(docker ps -q)
rm -f amdone.txt 
docker rmi -f flash:flattened_live
docker tag flash:flattened flash:flattened_live
echo "git config --global --add safe.directory '*' && python setup.py install && touch amdone.txt && echo done && ./loopforever.sh" | docker-compose -f docker-compose_compile.yaml run flashattention /bin/bash &
while [ ! -f ./amdone.txt ]
do
  sleep 0.2 # or less like 0.2
done
echo commiting container
docker commit $(docker ps -lq) flash:flattened_live
echo force-quitting docker
docker ps
docker kill $(docker ps -q)
docker ps
