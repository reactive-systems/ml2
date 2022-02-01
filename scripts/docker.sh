# Builds Docker images

CR="${ML2_CONTAINER_REGISTRY:-ml2}"

if [ $1 == "ml2" ]
then
    if [ $2 ]
    then
        docker build --build-arg CONTAINER_REGISTRY=$CR -t $CR:latest-$2 -f ../docker/ml2/$2.Dockerfile ..
    else
        for t in cpu gpu; do
            docker build --build-arg CONTAINER_REGISTRY=$CR -t $CR:latest-$t -f ../docker/ml2/$t.Dockerfile ..
        done
    fi
elif [ $1 == "deps" ]
then
    if [ $2 ]
    then 
        docker build --build-arg CONTAINER_REGISTRY=$CR -t $CR/$1:latest-$2 -f ../docker/$1/$2.Dockerfile ..
    else
        for t in cpu gpu; do
            docker build --build-arg CONTAINER_REGISTRY=$CR -t $CR/$1:latest-$t -f ../docker/$1/$t.Dockerfile ..
        done
    fi
elif [ $2 ]
then
    docker build --build-arg CONTAINER_REGISTRY=$CR -t $CR/$1:latest-$2 -f ../docker/$1/$2.Dockerfile ..
else
    docker build --build-arg CONTAINER_REGISTRY=$CR -t $CR/$1:latest -f ../docker/$1/Dockerfile ..
fi