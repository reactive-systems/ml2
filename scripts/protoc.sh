# Compiles protocol buffers files

AALTA_PROTO_PATH="../ml2/tools/aalta/aalta.proto"
BOSY_PROTO_PATH="../ml2/tools/bosy/bosy.proto"
LIMBOOLE_PROTO_PATH="../ml2/tools/limboole/limboole.proto"
LTL_PROTO_PATH="../ml2/tools/protos/ltl.proto"
NUXMV_PROTO_PATH="../ml2/tools/nuxmv/nuxmv.proto"
PROP_PROTO_PATH="../ml2/tools/protos/prop.proto"
SPOT_PROTO_PATH="../ml2/tools/spot/spot.proto"
STRIX_PROTO_PATH="../ml2/tools/strix/strix.proto"

if [ $1 == "aalta" ]
then
    PROTO_PATH=$AALTA_PROTO_PATH    
elif [ $1 == "bosy" ]
then
    PROTO_PATH=$BOSY_PROTO_PATH
elif [ $1 == "limboole" ]
then
    PROTO_PATH=$LIMBOOLE_PROTO_PATH
elif [ $1 == "ltl" ]
then
    PROTO_PATH=$LTL_PROTO_PATH
elif [ $1 == "nuxmv" ]
then
    PROTO_PATH=$NUXMV_PROTO_PATH
elif [ $1 == "prop" ]
then
    PROTO_PATH=$PROP_PROTO_PATH
elif [ $1 == "spot" ]
then
    PROTO_PATH=$SPOT_PROTO_PATH
elif [ $1 == "strix" ]
then
    PROTO_PATH=$STRIX_PROTO_PATH
else
    echo "Unknown argument"
    exit
fi

python -m grpc_tools.protoc --grpc_python_out=../ --python_out=../ --proto_path=../ $PROTO_PATH 