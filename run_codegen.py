from grpc_tools import protoc

protoc.main((
    '',
    '-Iproto',
    '--python_out=.',
    '--grpc_python_out=.',
    '--pyi_out=.',
    'proto/inference.proto',
))
