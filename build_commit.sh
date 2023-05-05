docker kill $(docker ps -q)
rm -f amdone.txt 
echo "python setup.py install && touch amdone.txt && echo done && ./loopforever.sh" | docker-compose run flashattention &
while [ ! -f ./amdone.txt ]
do
  sleep 0.2 # or less like 0.2
done
echo commiting container
docker commit $(docker ps -lq) flash:latest
echo force-quitting docker
docker ps
docker kill $(docker ps -q)
docker ps
