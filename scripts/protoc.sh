# Compiles protocol buffers files

GRPC_PATH="../ml2/grpc"

if [ $2 ]
then
    PROTO_PATH=$GRPC_PATH/$1/$2.proto
else
    PROTO_PATH=$GRPC_PATH/$1/$1.proto
fi

echo "Compiling protocol buffer at $PROTO_PATH"

python -m grpc_tools.protoc --grpc_python_out=../ --python_out=../ --proto_path=../ --mypy_out=../ $PROTO_PATH 