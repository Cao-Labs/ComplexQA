??-
?%?%
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
h
Any	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype?
?
BatchToSpaceND

input"T
block_shape"Tblock_shape
crops"Tcrops
output"T"	
Ttype" 
Tblock_shapetype0:
2	"
Tcropstype0:
2	
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
?
FloorMod
x"T
y"T
z"T"
Ttype:
2	
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
SpaceToBatchND

input"T
block_shape"Tblock_shape
paddings"	Tpaddings
output"T"	
Ttype" 
Tblock_shapetype0:
2	"
	Tpaddingstype0:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.22v2.6.1-9-gc2363d6d0258??'
?
"gcn_attention_30/conv1d_240/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"gcn_attention_30/conv1d_240/kernel
?
6gcn_attention_30/conv1d_240/kernel/Read/ReadVariableOpReadVariableOp"gcn_attention_30/conv1d_240/kernel*#
_output_shapes
:?*
dtype0
?
 gcn_attention_30/conv1d_240/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" gcn_attention_30/conv1d_240/bias
?
4gcn_attention_30/conv1d_240/bias/Read/ReadVariableOpReadVariableOp gcn_attention_30/conv1d_240/bias*
_output_shapes
:*
dtype0
?
.gcn_attention_30/batch_normalization_480/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.gcn_attention_30/batch_normalization_480/gamma
?
Bgcn_attention_30/batch_normalization_480/gamma/Read/ReadVariableOpReadVariableOp.gcn_attention_30/batch_normalization_480/gamma*
_output_shapes
:*
dtype0
?
-gcn_attention_30/batch_normalization_480/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-gcn_attention_30/batch_normalization_480/beta
?
Agcn_attention_30/batch_normalization_480/beta/Read/ReadVariableOpReadVariableOp-gcn_attention_30/batch_normalization_480/beta*
_output_shapes
:*
dtype0
?
4gcn_attention_30/batch_normalization_480/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64gcn_attention_30/batch_normalization_480/moving_mean
?
Hgcn_attention_30/batch_normalization_480/moving_mean/Read/ReadVariableOpReadVariableOp4gcn_attention_30/batch_normalization_480/moving_mean*
_output_shapes
:*
dtype0
?
8gcn_attention_30/batch_normalization_480/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8gcn_attention_30/batch_normalization_480/moving_variance
?
Lgcn_attention_30/batch_normalization_480/moving_variance/Read/ReadVariableOpReadVariableOp8gcn_attention_30/batch_normalization_480/moving_variance*
_output_shapes
:*
dtype0
?
 gcn_attention_30/dense_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*1
shared_name" gcn_attention_30/dense_30/kernel
?
4gcn_attention_30/dense_30/kernel/Read/ReadVariableOpReadVariableOp gcn_attention_30/dense_30/kernel*
_output_shapes

:	*
dtype0
?
"gcn_attention_30/conv2d_121/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*3
shared_name$"gcn_attention_30/conv2d_121/kernel
?
6gcn_attention_30/conv2d_121/kernel/Read/ReadVariableOpReadVariableOp"gcn_attention_30/conv2d_121/kernel*&
_output_shapes
:		*
dtype0
?
 gcn_attention_30/conv2d_121/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" gcn_attention_30/conv2d_121/bias
?
4gcn_attention_30/conv2d_121/bias/Read/ReadVariableOpReadVariableOp gcn_attention_30/conv2d_121/bias*
_output_shapes
:	*
dtype0
?
.gcn_attention_30/batch_normalization_482/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*?
shared_name0.gcn_attention_30/batch_normalization_482/gamma
?
Bgcn_attention_30/batch_normalization_482/gamma/Read/ReadVariableOpReadVariableOp.gcn_attention_30/batch_normalization_482/gamma*
_output_shapes
:	*
dtype0
?
-gcn_attention_30/batch_normalization_482/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*>
shared_name/-gcn_attention_30/batch_normalization_482/beta
?
Agcn_attention_30/batch_normalization_482/beta/Read/ReadVariableOpReadVariableOp-gcn_attention_30/batch_normalization_482/beta*
_output_shapes
:	*
dtype0
?
4gcn_attention_30/batch_normalization_482/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*E
shared_name64gcn_attention_30/batch_normalization_482/moving_mean
?
Hgcn_attention_30/batch_normalization_482/moving_mean/Read/ReadVariableOpReadVariableOp4gcn_attention_30/batch_normalization_482/moving_mean*
_output_shapes
:	*
dtype0
?
8gcn_attention_30/batch_normalization_482/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8gcn_attention_30/batch_normalization_482/moving_variance
?
Lgcn_attention_30/batch_normalization_482/moving_variance/Read/ReadVariableOpReadVariableOp8gcn_attention_30/batch_normalization_482/moving_variance*
_output_shapes
:	*
dtype0
?
(gcn_attention_30/embedding_30/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*9
shared_name*(gcn_attention_30/embedding_30/embeddings
?
<gcn_attention_30/embedding_30/embeddings/Read/ReadVariableOpReadVariableOp(gcn_attention_30/embedding_30/embeddings*
_output_shapes

:	*
dtype0
?
.gcn_attention_30/batch_normalization_483/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.gcn_attention_30/batch_normalization_483/gamma
?
Bgcn_attention_30/batch_normalization_483/gamma/Read/ReadVariableOpReadVariableOp.gcn_attention_30/batch_normalization_483/gamma*
_output_shapes
:*
dtype0
?
-gcn_attention_30/batch_normalization_483/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-gcn_attention_30/batch_normalization_483/beta
?
Agcn_attention_30/batch_normalization_483/beta/Read/ReadVariableOpReadVariableOp-gcn_attention_30/batch_normalization_483/beta*
_output_shapes
:*
dtype0
?
4gcn_attention_30/batch_normalization_483/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64gcn_attention_30/batch_normalization_483/moving_mean
?
Hgcn_attention_30/batch_normalization_483/moving_mean/Read/ReadVariableOpReadVariableOp4gcn_attention_30/batch_normalization_483/moving_mean*
_output_shapes
:*
dtype0
?
8gcn_attention_30/batch_normalization_483/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8gcn_attention_30/batch_normalization_483/moving_variance
?
Lgcn_attention_30/batch_normalization_483/moving_variance/Read/ReadVariableOpReadVariableOp8gcn_attention_30/batch_normalization_483/moving_variance*
_output_shapes
:*
dtype0
?
"gcn_attention_30/conv1d_247/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"gcn_attention_30/conv1d_247/kernel
?
6gcn_attention_30/conv1d_247/kernel/Read/ReadVariableOpReadVariableOp"gcn_attention_30/conv1d_247/kernel*"
_output_shapes
:*
dtype0
?
 gcn_attention_30/conv1d_247/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" gcn_attention_30/conv1d_247/bias
?
4gcn_attention_30/conv1d_247/bias/Read/ReadVariableOpReadVariableOp gcn_attention_30/conv1d_247/bias*
_output_shapes
:*
dtype0
?
.gcn_attention_30/core_gcn_60/conv2d_122/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.gcn_attention_30/core_gcn_60/conv2d_122/kernel
?
Bgcn_attention_30/core_gcn_60/conv2d_122/kernel/Read/ReadVariableOpReadVariableOp.gcn_attention_30/core_gcn_60/conv2d_122/kernel*&
_output_shapes
:*
dtype0
?
,gcn_attention_30/core_gcn_60/conv2d_122/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,gcn_attention_30/core_gcn_60/conv2d_122/bias
?
@gcn_attention_30/core_gcn_60/conv2d_122/bias/Read/ReadVariableOpReadVariableOp,gcn_attention_30/core_gcn_60/conv2d_122/bias*
_output_shapes
:*
dtype0
?
:gcn_attention_30/core_gcn_60/batch_normalization_484/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_30/core_gcn_60/batch_normalization_484/gamma
?
Ngcn_attention_30/core_gcn_60/batch_normalization_484/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_30/core_gcn_60/batch_normalization_484/gamma*
_output_shapes
:*
dtype0
?
9gcn_attention_30/core_gcn_60/batch_normalization_484/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_30/core_gcn_60/batch_normalization_484/beta
?
Mgcn_attention_30/core_gcn_60/batch_normalization_484/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_30/core_gcn_60/batch_normalization_484/beta*
_output_shapes
:*
dtype0
?
.gcn_attention_30/core_gcn_60/conv1d_241/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.gcn_attention_30/core_gcn_60/conv1d_241/kernel
?
Bgcn_attention_30/core_gcn_60/conv1d_241/kernel/Read/ReadVariableOpReadVariableOp.gcn_attention_30/core_gcn_60/conv1d_241/kernel*"
_output_shapes
:*
dtype0
?
,gcn_attention_30/core_gcn_60/conv1d_241/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,gcn_attention_30/core_gcn_60/conv1d_241/bias
?
@gcn_attention_30/core_gcn_60/conv1d_241/bias/Read/ReadVariableOpReadVariableOp,gcn_attention_30/core_gcn_60/conv1d_241/bias*
_output_shapes
:*
dtype0
?
:gcn_attention_30/core_gcn_60/batch_normalization_485/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_30/core_gcn_60/batch_normalization_485/gamma
?
Ngcn_attention_30/core_gcn_60/batch_normalization_485/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_30/core_gcn_60/batch_normalization_485/gamma*
_output_shapes
:*
dtype0
?
9gcn_attention_30/core_gcn_60/batch_normalization_485/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_30/core_gcn_60/batch_normalization_485/beta
?
Mgcn_attention_30/core_gcn_60/batch_normalization_485/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_30/core_gcn_60/batch_normalization_485/beta*
_output_shapes
:*
dtype0
?
.gcn_attention_30/core_gcn_60/conv1d_242/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.gcn_attention_30/core_gcn_60/conv1d_242/kernel
?
Bgcn_attention_30/core_gcn_60/conv1d_242/kernel/Read/ReadVariableOpReadVariableOp.gcn_attention_30/core_gcn_60/conv1d_242/kernel*"
_output_shapes
:*
dtype0
?
,gcn_attention_30/core_gcn_60/conv1d_242/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,gcn_attention_30/core_gcn_60/conv1d_242/bias
?
@gcn_attention_30/core_gcn_60/conv1d_242/bias/Read/ReadVariableOpReadVariableOp,gcn_attention_30/core_gcn_60/conv1d_242/bias*
_output_shapes
:*
dtype0
?
:gcn_attention_30/core_gcn_60/batch_normalization_486/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_30/core_gcn_60/batch_normalization_486/gamma
?
Ngcn_attention_30/core_gcn_60/batch_normalization_486/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_30/core_gcn_60/batch_normalization_486/gamma*
_output_shapes
:*
dtype0
?
9gcn_attention_30/core_gcn_60/batch_normalization_486/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_30/core_gcn_60/batch_normalization_486/beta
?
Mgcn_attention_30/core_gcn_60/batch_normalization_486/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_30/core_gcn_60/batch_normalization_486/beta*
_output_shapes
:*
dtype0
?
.gcn_attention_30/core_gcn_60/conv1d_243/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.gcn_attention_30/core_gcn_60/conv1d_243/kernel
?
Bgcn_attention_30/core_gcn_60/conv1d_243/kernel/Read/ReadVariableOpReadVariableOp.gcn_attention_30/core_gcn_60/conv1d_243/kernel*"
_output_shapes
:*
dtype0
?
,gcn_attention_30/core_gcn_60/conv1d_243/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,gcn_attention_30/core_gcn_60/conv1d_243/bias
?
@gcn_attention_30/core_gcn_60/conv1d_243/bias/Read/ReadVariableOpReadVariableOp,gcn_attention_30/core_gcn_60/conv1d_243/bias*
_output_shapes
:*
dtype0
?
:gcn_attention_30/core_gcn_60/batch_normalization_487/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_30/core_gcn_60/batch_normalization_487/gamma
?
Ngcn_attention_30/core_gcn_60/batch_normalization_487/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_30/core_gcn_60/batch_normalization_487/gamma*
_output_shapes
:*
dtype0
?
9gcn_attention_30/core_gcn_60/batch_normalization_487/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_30/core_gcn_60/batch_normalization_487/beta
?
Mgcn_attention_30/core_gcn_60/batch_normalization_487/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_30/core_gcn_60/batch_normalization_487/beta*
_output_shapes
:*
dtype0
?
:gcn_attention_30/core_gcn_60/batch_normalization_488/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_30/core_gcn_60/batch_normalization_488/gamma
?
Ngcn_attention_30/core_gcn_60/batch_normalization_488/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_30/core_gcn_60/batch_normalization_488/gamma*
_output_shapes
:*
dtype0
?
9gcn_attention_30/core_gcn_60/batch_normalization_488/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_30/core_gcn_60/batch_normalization_488/beta
?
Mgcn_attention_30/core_gcn_60/batch_normalization_488/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_30/core_gcn_60/batch_normalization_488/beta*
_output_shapes
:*
dtype0
?
:gcn_attention_30/core_gcn_60/batch_normalization_489/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_30/core_gcn_60/batch_normalization_489/gamma
?
Ngcn_attention_30/core_gcn_60/batch_normalization_489/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_30/core_gcn_60/batch_normalization_489/gamma*
_output_shapes
:*
dtype0
?
9gcn_attention_30/core_gcn_60/batch_normalization_489/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_30/core_gcn_60/batch_normalization_489/beta
?
Mgcn_attention_30/core_gcn_60/batch_normalization_489/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_30/core_gcn_60/batch_normalization_489/beta*
_output_shapes
:*
dtype0
?
.gcn_attention_30/core_gcn_61/conv2d_123/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.gcn_attention_30/core_gcn_61/conv2d_123/kernel
?
Bgcn_attention_30/core_gcn_61/conv2d_123/kernel/Read/ReadVariableOpReadVariableOp.gcn_attention_30/core_gcn_61/conv2d_123/kernel*&
_output_shapes
:*
dtype0
?
,gcn_attention_30/core_gcn_61/conv2d_123/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,gcn_attention_30/core_gcn_61/conv2d_123/bias
?
@gcn_attention_30/core_gcn_61/conv2d_123/bias/Read/ReadVariableOpReadVariableOp,gcn_attention_30/core_gcn_61/conv2d_123/bias*
_output_shapes
:*
dtype0
?
:gcn_attention_30/core_gcn_61/batch_normalization_490/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_30/core_gcn_61/batch_normalization_490/gamma
?
Ngcn_attention_30/core_gcn_61/batch_normalization_490/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_30/core_gcn_61/batch_normalization_490/gamma*
_output_shapes
:*
dtype0
?
9gcn_attention_30/core_gcn_61/batch_normalization_490/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_30/core_gcn_61/batch_normalization_490/beta
?
Mgcn_attention_30/core_gcn_61/batch_normalization_490/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_30/core_gcn_61/batch_normalization_490/beta*
_output_shapes
:*
dtype0
?
.gcn_attention_30/core_gcn_61/conv1d_244/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.gcn_attention_30/core_gcn_61/conv1d_244/kernel
?
Bgcn_attention_30/core_gcn_61/conv1d_244/kernel/Read/ReadVariableOpReadVariableOp.gcn_attention_30/core_gcn_61/conv1d_244/kernel*"
_output_shapes
:*
dtype0
?
,gcn_attention_30/core_gcn_61/conv1d_244/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,gcn_attention_30/core_gcn_61/conv1d_244/bias
?
@gcn_attention_30/core_gcn_61/conv1d_244/bias/Read/ReadVariableOpReadVariableOp,gcn_attention_30/core_gcn_61/conv1d_244/bias*
_output_shapes
:*
dtype0
?
:gcn_attention_30/core_gcn_61/batch_normalization_491/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_30/core_gcn_61/batch_normalization_491/gamma
?
Ngcn_attention_30/core_gcn_61/batch_normalization_491/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_30/core_gcn_61/batch_normalization_491/gamma*
_output_shapes
:*
dtype0
?
9gcn_attention_30/core_gcn_61/batch_normalization_491/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_30/core_gcn_61/batch_normalization_491/beta
?
Mgcn_attention_30/core_gcn_61/batch_normalization_491/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_30/core_gcn_61/batch_normalization_491/beta*
_output_shapes
:*
dtype0
?
.gcn_attention_30/core_gcn_61/conv1d_245/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.gcn_attention_30/core_gcn_61/conv1d_245/kernel
?
Bgcn_attention_30/core_gcn_61/conv1d_245/kernel/Read/ReadVariableOpReadVariableOp.gcn_attention_30/core_gcn_61/conv1d_245/kernel*"
_output_shapes
:*
dtype0
?
,gcn_attention_30/core_gcn_61/conv1d_245/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,gcn_attention_30/core_gcn_61/conv1d_245/bias
?
@gcn_attention_30/core_gcn_61/conv1d_245/bias/Read/ReadVariableOpReadVariableOp,gcn_attention_30/core_gcn_61/conv1d_245/bias*
_output_shapes
:*
dtype0
?
:gcn_attention_30/core_gcn_61/batch_normalization_492/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_30/core_gcn_61/batch_normalization_492/gamma
?
Ngcn_attention_30/core_gcn_61/batch_normalization_492/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_30/core_gcn_61/batch_normalization_492/gamma*
_output_shapes
:*
dtype0
?
9gcn_attention_30/core_gcn_61/batch_normalization_492/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_30/core_gcn_61/batch_normalization_492/beta
?
Mgcn_attention_30/core_gcn_61/batch_normalization_492/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_30/core_gcn_61/batch_normalization_492/beta*
_output_shapes
:*
dtype0
?
.gcn_attention_30/core_gcn_61/conv1d_246/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.gcn_attention_30/core_gcn_61/conv1d_246/kernel
?
Bgcn_attention_30/core_gcn_61/conv1d_246/kernel/Read/ReadVariableOpReadVariableOp.gcn_attention_30/core_gcn_61/conv1d_246/kernel*"
_output_shapes
:*
dtype0
?
,gcn_attention_30/core_gcn_61/conv1d_246/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,gcn_attention_30/core_gcn_61/conv1d_246/bias
?
@gcn_attention_30/core_gcn_61/conv1d_246/bias/Read/ReadVariableOpReadVariableOp,gcn_attention_30/core_gcn_61/conv1d_246/bias*
_output_shapes
:*
dtype0
?
:gcn_attention_30/core_gcn_61/batch_normalization_493/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_30/core_gcn_61/batch_normalization_493/gamma
?
Ngcn_attention_30/core_gcn_61/batch_normalization_493/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_30/core_gcn_61/batch_normalization_493/gamma*
_output_shapes
:*
dtype0
?
9gcn_attention_30/core_gcn_61/batch_normalization_493/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_30/core_gcn_61/batch_normalization_493/beta
?
Mgcn_attention_30/core_gcn_61/batch_normalization_493/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_30/core_gcn_61/batch_normalization_493/beta*
_output_shapes
:*
dtype0
?
:gcn_attention_30/core_gcn_61/batch_normalization_494/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_30/core_gcn_61/batch_normalization_494/gamma
?
Ngcn_attention_30/core_gcn_61/batch_normalization_494/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_30/core_gcn_61/batch_normalization_494/gamma*
_output_shapes
:*
dtype0
?
9gcn_attention_30/core_gcn_61/batch_normalization_494/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_30/core_gcn_61/batch_normalization_494/beta
?
Mgcn_attention_30/core_gcn_61/batch_normalization_494/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_30/core_gcn_61/batch_normalization_494/beta*
_output_shapes
:*
dtype0
?
:gcn_attention_30/core_gcn_61/batch_normalization_495/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_30/core_gcn_61/batch_normalization_495/gamma
?
Ngcn_attention_30/core_gcn_61/batch_normalization_495/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_30/core_gcn_61/batch_normalization_495/gamma*
_output_shapes
:*
dtype0
?
9gcn_attention_30/core_gcn_61/batch_normalization_495/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_30/core_gcn_61/batch_normalization_495/beta
?
Mgcn_attention_30/core_gcn_61/batch_normalization_495/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_30/core_gcn_61/batch_normalization_495/beta*
_output_shapes
:*
dtype0
?
@gcn_attention_30/core_gcn_60/batch_normalization_484/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_30/core_gcn_60/batch_normalization_484/moving_mean
?
Tgcn_attention_30/core_gcn_60/batch_normalization_484/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_30/core_gcn_60/batch_normalization_484/moving_mean*
_output_shapes
:*
dtype0
?
Dgcn_attention_30/core_gcn_60/batch_normalization_484/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_30/core_gcn_60/batch_normalization_484/moving_variance
?
Xgcn_attention_30/core_gcn_60/batch_normalization_484/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_30/core_gcn_60/batch_normalization_484/moving_variance*
_output_shapes
:*
dtype0
?
@gcn_attention_30/core_gcn_60/batch_normalization_485/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_30/core_gcn_60/batch_normalization_485/moving_mean
?
Tgcn_attention_30/core_gcn_60/batch_normalization_485/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_30/core_gcn_60/batch_normalization_485/moving_mean*
_output_shapes
:*
dtype0
?
Dgcn_attention_30/core_gcn_60/batch_normalization_485/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_30/core_gcn_60/batch_normalization_485/moving_variance
?
Xgcn_attention_30/core_gcn_60/batch_normalization_485/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_30/core_gcn_60/batch_normalization_485/moving_variance*
_output_shapes
:*
dtype0
?
@gcn_attention_30/core_gcn_60/batch_normalization_486/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_30/core_gcn_60/batch_normalization_486/moving_mean
?
Tgcn_attention_30/core_gcn_60/batch_normalization_486/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_30/core_gcn_60/batch_normalization_486/moving_mean*
_output_shapes
:*
dtype0
?
Dgcn_attention_30/core_gcn_60/batch_normalization_486/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_30/core_gcn_60/batch_normalization_486/moving_variance
?
Xgcn_attention_30/core_gcn_60/batch_normalization_486/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_30/core_gcn_60/batch_normalization_486/moving_variance*
_output_shapes
:*
dtype0
?
@gcn_attention_30/core_gcn_60/batch_normalization_487/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_30/core_gcn_60/batch_normalization_487/moving_mean
?
Tgcn_attention_30/core_gcn_60/batch_normalization_487/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_30/core_gcn_60/batch_normalization_487/moving_mean*
_output_shapes
:*
dtype0
?
Dgcn_attention_30/core_gcn_60/batch_normalization_487/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_30/core_gcn_60/batch_normalization_487/moving_variance
?
Xgcn_attention_30/core_gcn_60/batch_normalization_487/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_30/core_gcn_60/batch_normalization_487/moving_variance*
_output_shapes
:*
dtype0
?
@gcn_attention_30/core_gcn_60/batch_normalization_488/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_30/core_gcn_60/batch_normalization_488/moving_mean
?
Tgcn_attention_30/core_gcn_60/batch_normalization_488/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_30/core_gcn_60/batch_normalization_488/moving_mean*
_output_shapes
:*
dtype0
?
Dgcn_attention_30/core_gcn_60/batch_normalization_488/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_30/core_gcn_60/batch_normalization_488/moving_variance
?
Xgcn_attention_30/core_gcn_60/batch_normalization_488/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_30/core_gcn_60/batch_normalization_488/moving_variance*
_output_shapes
:*
dtype0
?
@gcn_attention_30/core_gcn_60/batch_normalization_489/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_30/core_gcn_60/batch_normalization_489/moving_mean
?
Tgcn_attention_30/core_gcn_60/batch_normalization_489/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_30/core_gcn_60/batch_normalization_489/moving_mean*
_output_shapes
:*
dtype0
?
Dgcn_attention_30/core_gcn_60/batch_normalization_489/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_30/core_gcn_60/batch_normalization_489/moving_variance
?
Xgcn_attention_30/core_gcn_60/batch_normalization_489/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_30/core_gcn_60/batch_normalization_489/moving_variance*
_output_shapes
:*
dtype0
?
@gcn_attention_30/core_gcn_61/batch_normalization_490/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_30/core_gcn_61/batch_normalization_490/moving_mean
?
Tgcn_attention_30/core_gcn_61/batch_normalization_490/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_30/core_gcn_61/batch_normalization_490/moving_mean*
_output_shapes
:*
dtype0
?
Dgcn_attention_30/core_gcn_61/batch_normalization_490/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_30/core_gcn_61/batch_normalization_490/moving_variance
?
Xgcn_attention_30/core_gcn_61/batch_normalization_490/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_30/core_gcn_61/batch_normalization_490/moving_variance*
_output_shapes
:*
dtype0
?
@gcn_attention_30/core_gcn_61/batch_normalization_491/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_30/core_gcn_61/batch_normalization_491/moving_mean
?
Tgcn_attention_30/core_gcn_61/batch_normalization_491/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_30/core_gcn_61/batch_normalization_491/moving_mean*
_output_shapes
:*
dtype0
?
Dgcn_attention_30/core_gcn_61/batch_normalization_491/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_30/core_gcn_61/batch_normalization_491/moving_variance
?
Xgcn_attention_30/core_gcn_61/batch_normalization_491/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_30/core_gcn_61/batch_normalization_491/moving_variance*
_output_shapes
:*
dtype0
?
@gcn_attention_30/core_gcn_61/batch_normalization_492/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_30/core_gcn_61/batch_normalization_492/moving_mean
?
Tgcn_attention_30/core_gcn_61/batch_normalization_492/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_30/core_gcn_61/batch_normalization_492/moving_mean*
_output_shapes
:*
dtype0
?
Dgcn_attention_30/core_gcn_61/batch_normalization_492/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_30/core_gcn_61/batch_normalization_492/moving_variance
?
Xgcn_attention_30/core_gcn_61/batch_normalization_492/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_30/core_gcn_61/batch_normalization_492/moving_variance*
_output_shapes
:*
dtype0
?
@gcn_attention_30/core_gcn_61/batch_normalization_493/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_30/core_gcn_61/batch_normalization_493/moving_mean
?
Tgcn_attention_30/core_gcn_61/batch_normalization_493/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_30/core_gcn_61/batch_normalization_493/moving_mean*
_output_shapes
:*
dtype0
?
Dgcn_attention_30/core_gcn_61/batch_normalization_493/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_30/core_gcn_61/batch_normalization_493/moving_variance
?
Xgcn_attention_30/core_gcn_61/batch_normalization_493/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_30/core_gcn_61/batch_normalization_493/moving_variance*
_output_shapes
:*
dtype0
?
@gcn_attention_30/core_gcn_61/batch_normalization_494/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_30/core_gcn_61/batch_normalization_494/moving_mean
?
Tgcn_attention_30/core_gcn_61/batch_normalization_494/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_30/core_gcn_61/batch_normalization_494/moving_mean*
_output_shapes
:*
dtype0
?
Dgcn_attention_30/core_gcn_61/batch_normalization_494/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_30/core_gcn_61/batch_normalization_494/moving_variance
?
Xgcn_attention_30/core_gcn_61/batch_normalization_494/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_30/core_gcn_61/batch_normalization_494/moving_variance*
_output_shapes
:*
dtype0
?
@gcn_attention_30/core_gcn_61/batch_normalization_495/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_30/core_gcn_61/batch_normalization_495/moving_mean
?
Tgcn_attention_30/core_gcn_61/batch_normalization_495/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_30/core_gcn_61/batch_normalization_495/moving_mean*
_output_shapes
:*
dtype0
?
Dgcn_attention_30/core_gcn_61/batch_normalization_495/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_30/core_gcn_61/batch_normalization_495/moving_variance
?
Xgcn_attention_30/core_gcn_61/batch_normalization_495/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_30/core_gcn_61/batch_normalization_495/moving_variance*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
*
self_attention_layer

signatures
?
node0_Conv1D
node0_Activation
node0_BatchNormalization
node0_Dropout
edge_Convolution2D
edge_Activation
	edge_BatchNormalization

edge_Dropout
distance_Dense
distance_Convolution2D
distance_Activation
distance_BatchNormalization
distance_Dropout
	adj_Dense
node_Activation5
node_BatchNormalization5
node_Dropout5

gcn_layers
Activation_sig
node_Conv1D_out
relu_out
	variables
trainable_variables
regularization_losses
	keras_api
 
h

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
R
"	variables
#trainable_variables
$regularization_losses
%	keras_api
?
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+	variables
,trainable_variables
-regularization_losses
.	keras_api
R
/	variables
0trainable_variables
1regularization_losses
2	keras_api

3	keras_api

4	keras_api

5	keras_api

6	keras_api
^

7kernel
8	variables
9trainable_variables
:regularization_losses
;	keras_api
h

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
R
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
?
Faxis
	Ggamma
Hbeta
Imoving_mean
Jmoving_variance
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
R
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
b
S
embeddings
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
R
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
?
\axis
	]gamma
^beta
_moving_mean
`moving_variance
a	variables
btrainable_variables
cregularization_losses
d	keras_api
R
e	variables
ftrainable_variables
gregularization_losses
h	keras_api

i0
j1

k	keras_api
h

lkernel
mbias
n	variables
otrainable_variables
pregularization_losses
q	keras_api
R
r	variables
strainable_variables
tregularization_losses
u	keras_api
?
0
1
'2
(3
74
<5
=6
G7
H8
S9
]10
^11
v12
w13
x14
y15
z16
{17
|18
}19
~20
21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
l52
m53
)54
*55
I56
J57
_58
`59
?60
?61
?62
?63
?64
?65
?66
?67
?68
?69
?70
?71
?72
?73
?74
?75
?76
?77
?78
?79
?80
?81
?82
?83
?
0
1
'2
(3
74
<5
=6
G7
H8
S9
]10
^11
v12
w13
x14
y15
z16
{17
|18
}19
~20
21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
l52
m53
 
?
	variables
?non_trainable_variables
trainable_variables
?layers
regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
{y
VARIABLE_VALUE"gcn_attention_30/conv1d_240/kernelCself_attention_layer/node0_Conv1D/kernel/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE gcn_attention_30/conv1d_240/biasAself_attention_layer/node0_Conv1D/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables
?non_trainable_variables
trainable_variables
?layers
 regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
"	variables
?non_trainable_variables
#trainable_variables
?layers
$regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
??
VARIABLE_VALUE.gcn_attention_30/batch_normalization_480/gammaNself_attention_layer/node0_BatchNormalization/gamma/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-gcn_attention_30/batch_normalization_480/betaMself_attention_layer/node0_BatchNormalization/beta/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4gcn_attention_30/batch_normalization_480/moving_meanTself_attention_layer/node0_BatchNormalization/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8gcn_attention_30/batch_normalization_480/moving_varianceXself_attention_layer/node0_BatchNormalization/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

'0
(1
)2
*3

'0
(1
 
?
+	variables
?non_trainable_variables
,trainable_variables
?layers
-regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
/	variables
?non_trainable_variables
0trainable_variables
?layers
1regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
 
{y
VARIABLE_VALUE gcn_attention_30/dense_30/kernelEself_attention_layer/distance_Dense/kernel/.ATTRIBUTES/VARIABLE_VALUE

70

70
 
?
8	variables
?non_trainable_variables
9trainable_variables
?layers
:regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
??
VARIABLE_VALUE"gcn_attention_30/conv2d_121/kernelMself_attention_layer/distance_Convolution2D/kernel/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE gcn_attention_30/conv2d_121/biasKself_attention_layer/distance_Convolution2D/bias/.ATTRIBUTES/VARIABLE_VALUE

<0
=1

<0
=1
 
?
>	variables
?non_trainable_variables
?trainable_variables
?layers
@regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
B	variables
?non_trainable_variables
Ctrainable_variables
?layers
Dregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
??
VARIABLE_VALUE.gcn_attention_30/batch_normalization_482/gammaQself_attention_layer/distance_BatchNormalization/gamma/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-gcn_attention_30/batch_normalization_482/betaPself_attention_layer/distance_BatchNormalization/beta/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4gcn_attention_30/batch_normalization_482/moving_meanWself_attention_layer/distance_BatchNormalization/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8gcn_attention_30/batch_normalization_482/moving_variance[self_attention_layer/distance_BatchNormalization/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

G0
H1
I2
J3

G0
H1
 
?
K	variables
?non_trainable_variables
Ltrainable_variables
?layers
Mregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
O	variables
?non_trainable_variables
Ptrainable_variables
?layers
Qregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
??
VARIABLE_VALUE(gcn_attention_30/embedding_30/embeddingsDself_attention_layer/adj_Dense/embeddings/.ATTRIBUTES/VARIABLE_VALUE

S0

S0
 
?
T	variables
?non_trainable_variables
Utrainable_variables
?layers
Vregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
X	variables
?non_trainable_variables
Ytrainable_variables
?layers
Zregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
??
VARIABLE_VALUE.gcn_attention_30/batch_normalization_483/gammaNself_attention_layer/node_BatchNormalization5/gamma/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-gcn_attention_30/batch_normalization_483/betaMself_attention_layer/node_BatchNormalization5/beta/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4gcn_attention_30/batch_normalization_483/moving_meanTself_attention_layer/node_BatchNormalization5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8gcn_attention_30/batch_normalization_483/moving_varianceXself_attention_layer/node_BatchNormalization5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

]0
^1
_2
`3

]0
^1
 
?
a	variables
?non_trainable_variables
btrainable_variables
?layers
cregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
e	variables
?non_trainable_variables
ftrainable_variables
?layers
gregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?
?conv2d_1
?batch_norm_1
?	dropout_1
?conv1d_1
?batch_norm_2
?	dropout_2
?conv1d_2
?batch_norm_3
?	dropout_3
?conv1d_3
?batch_norm_4
?	dropout_4
?batch_norm_5
?batch_norm_6
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
?conv2d_1
?batch_norm_1
?	dropout_1
?conv1d_1
?batch_norm_2
?	dropout_2
?conv1d_2
?batch_norm_3
?	dropout_3
?conv1d_3
?batch_norm_4
?	dropout_4
?batch_norm_5
?batch_norm_6
?	variables
?trainable_variables
?regularization_losses
?	keras_api
 
~|
VARIABLE_VALUE"gcn_attention_30/conv1d_247/kernelFself_attention_layer/node_Conv1D_out/kernel/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE gcn_attention_30/conv1d_247/biasDself_attention_layer/node_Conv1D_out/bias/.ATTRIBUTES/VARIABLE_VALUE

l0
m1

l0
m1
 
?
n	variables
?non_trainable_variables
otrainable_variables
?layers
pregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
r	variables
?non_trainable_variables
strainable_variables
?layers
tregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?~
VARIABLE_VALUE.gcn_attention_30/core_gcn_60/conv2d_122/kernel<self_attention_layer/variables/12/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE,gcn_attention_30/core_gcn_60/conv2d_122/bias<self_attention_layer/variables/13/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE:gcn_attention_30/core_gcn_60/batch_normalization_484/gamma<self_attention_layer/variables/14/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9gcn_attention_30/core_gcn_60/batch_normalization_484/beta<self_attention_layer/variables/15/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE.gcn_attention_30/core_gcn_60/conv1d_241/kernel<self_attention_layer/variables/16/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE,gcn_attention_30/core_gcn_60/conv1d_241/bias<self_attention_layer/variables/17/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE:gcn_attention_30/core_gcn_60/batch_normalization_485/gamma<self_attention_layer/variables/18/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9gcn_attention_30/core_gcn_60/batch_normalization_485/beta<self_attention_layer/variables/19/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE.gcn_attention_30/core_gcn_60/conv1d_242/kernel<self_attention_layer/variables/20/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE,gcn_attention_30/core_gcn_60/conv1d_242/bias<self_attention_layer/variables/21/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE:gcn_attention_30/core_gcn_60/batch_normalization_486/gamma<self_attention_layer/variables/22/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9gcn_attention_30/core_gcn_60/batch_normalization_486/beta<self_attention_layer/variables/23/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE.gcn_attention_30/core_gcn_60/conv1d_243/kernel<self_attention_layer/variables/24/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE,gcn_attention_30/core_gcn_60/conv1d_243/bias<self_attention_layer/variables/25/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE:gcn_attention_30/core_gcn_60/batch_normalization_487/gamma<self_attention_layer/variables/26/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9gcn_attention_30/core_gcn_60/batch_normalization_487/beta<self_attention_layer/variables/27/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE:gcn_attention_30/core_gcn_60/batch_normalization_488/gamma<self_attention_layer/variables/28/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9gcn_attention_30/core_gcn_60/batch_normalization_488/beta<self_attention_layer/variables/29/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE:gcn_attention_30/core_gcn_60/batch_normalization_489/gamma<self_attention_layer/variables/30/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9gcn_attention_30/core_gcn_60/batch_normalization_489/beta<self_attention_layer/variables/31/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE.gcn_attention_30/core_gcn_61/conv2d_123/kernel<self_attention_layer/variables/32/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE,gcn_attention_30/core_gcn_61/conv2d_123/bias<self_attention_layer/variables/33/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE:gcn_attention_30/core_gcn_61/batch_normalization_490/gamma<self_attention_layer/variables/34/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9gcn_attention_30/core_gcn_61/batch_normalization_490/beta<self_attention_layer/variables/35/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE.gcn_attention_30/core_gcn_61/conv1d_244/kernel<self_attention_layer/variables/36/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE,gcn_attention_30/core_gcn_61/conv1d_244/bias<self_attention_layer/variables/37/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE:gcn_attention_30/core_gcn_61/batch_normalization_491/gamma<self_attention_layer/variables/38/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9gcn_attention_30/core_gcn_61/batch_normalization_491/beta<self_attention_layer/variables/39/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE.gcn_attention_30/core_gcn_61/conv1d_245/kernel<self_attention_layer/variables/40/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE,gcn_attention_30/core_gcn_61/conv1d_245/bias<self_attention_layer/variables/41/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE:gcn_attention_30/core_gcn_61/batch_normalization_492/gamma<self_attention_layer/variables/42/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9gcn_attention_30/core_gcn_61/batch_normalization_492/beta<self_attention_layer/variables/43/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE.gcn_attention_30/core_gcn_61/conv1d_246/kernel<self_attention_layer/variables/44/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE,gcn_attention_30/core_gcn_61/conv1d_246/bias<self_attention_layer/variables/45/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE:gcn_attention_30/core_gcn_61/batch_normalization_493/gamma<self_attention_layer/variables/46/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9gcn_attention_30/core_gcn_61/batch_normalization_493/beta<self_attention_layer/variables/47/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE:gcn_attention_30/core_gcn_61/batch_normalization_494/gamma<self_attention_layer/variables/48/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9gcn_attention_30/core_gcn_61/batch_normalization_494/beta<self_attention_layer/variables/49/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE:gcn_attention_30/core_gcn_61/batch_normalization_495/gamma<self_attention_layer/variables/50/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9gcn_attention_30/core_gcn_61/batch_normalization_495/beta<self_attention_layer/variables/51/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@gcn_attention_30/core_gcn_60/batch_normalization_484/moving_mean<self_attention_layer/variables/60/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDgcn_attention_30/core_gcn_60/batch_normalization_484/moving_variance<self_attention_layer/variables/61/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@gcn_attention_30/core_gcn_60/batch_normalization_485/moving_mean<self_attention_layer/variables/62/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDgcn_attention_30/core_gcn_60/batch_normalization_485/moving_variance<self_attention_layer/variables/63/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@gcn_attention_30/core_gcn_60/batch_normalization_486/moving_mean<self_attention_layer/variables/64/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDgcn_attention_30/core_gcn_60/batch_normalization_486/moving_variance<self_attention_layer/variables/65/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@gcn_attention_30/core_gcn_60/batch_normalization_487/moving_mean<self_attention_layer/variables/66/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDgcn_attention_30/core_gcn_60/batch_normalization_487/moving_variance<self_attention_layer/variables/67/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@gcn_attention_30/core_gcn_60/batch_normalization_488/moving_mean<self_attention_layer/variables/68/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDgcn_attention_30/core_gcn_60/batch_normalization_488/moving_variance<self_attention_layer/variables/69/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@gcn_attention_30/core_gcn_60/batch_normalization_489/moving_mean<self_attention_layer/variables/70/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDgcn_attention_30/core_gcn_60/batch_normalization_489/moving_variance<self_attention_layer/variables/71/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@gcn_attention_30/core_gcn_61/batch_normalization_490/moving_mean<self_attention_layer/variables/72/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDgcn_attention_30/core_gcn_61/batch_normalization_490/moving_variance<self_attention_layer/variables/73/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@gcn_attention_30/core_gcn_61/batch_normalization_491/moving_mean<self_attention_layer/variables/74/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDgcn_attention_30/core_gcn_61/batch_normalization_491/moving_variance<self_attention_layer/variables/75/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@gcn_attention_30/core_gcn_61/batch_normalization_492/moving_mean<self_attention_layer/variables/76/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDgcn_attention_30/core_gcn_61/batch_normalization_492/moving_variance<self_attention_layer/variables/77/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@gcn_attention_30/core_gcn_61/batch_normalization_493/moving_mean<self_attention_layer/variables/78/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDgcn_attention_30/core_gcn_61/batch_normalization_493/moving_variance<self_attention_layer/variables/79/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@gcn_attention_30/core_gcn_61/batch_normalization_494/moving_mean<self_attention_layer/variables/80/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDgcn_attention_30/core_gcn_61/batch_normalization_494/moving_variance<self_attention_layer/variables/81/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@gcn_attention_30/core_gcn_61/batch_normalization_495/moving_mean<self_attention_layer/variables/82/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDgcn_attention_30/core_gcn_61/batch_normalization_495/moving_variance<self_attention_layer/variables/83/.ATTRIBUTES/VARIABLE_VALUE
?
)0
*1
I2
J3
_4
`5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?
0
1
2
3
4
5
	6

7
8
9
10
11
12
13
14
15
16
i17
j18
19
20
21
 
 
 
 
 
 
 
 
 
 
 
 
 

)0
*1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

I0
J1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

_0
`1
 
 
 
 
 
 
 
 
 
l

vkernel
wbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	xgamma
ybeta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

zkernel
{bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	|gamma
}beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

~kernel
bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
v0
w1
x2
y3
z4
{5
|6
}7
~8
9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?
v0
w1
x2
y3
z4
{5
|6
}7
~8
9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
 
 
 
 
 
 
 

v0
w1

v0
w1
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 

x0
y1
?2
?3

x0
y1
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses

z0
{1

z0
{1
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 

|0
}1
?2
?3

|0
}1
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses

~0
1

~0
1
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
?0
?1
?2
?3

?0
?1
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses

?0
?1

?0
?1
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
?0
?1
?2
?3

?0
?1
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
?0
?1
?2
?3

?0
?1
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
?0
?1
?2
?3

?0
?1
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
b
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
t
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
 
 
 

?0
?1

?0
?1
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
?0
?1
?2
?3

?0
?1
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses

?0
?1

?0
?1
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
?0
?1
?2
?3

?0
?1
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses

?0
?1

?0
?1
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
?0
?1
?2
?3

?0
?1
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses

?0
?1

?0
?1
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
?0
?1
?2
?3

?0
?1
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
?0
?1
?2
?3

?0
?1
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
 
 
?0
?1
?2
?3

?0
?1
 
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
b
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
t
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 

?0
?1
 
 
 
 
?
serving_default_input_text1Placeholder*5
_output_shapes#
!:???????????????????*
dtype0**
shape!:???????????????????
?
serving_default_input_text2Placeholder*A
_output_shapes/
-:+???????????????????????????*
dtype0*6
shape-:+???????????????????????????
?
serving_default_input_text3Placeholder*A
_output_shapes/
-:+???????????????????????????*
dtype0*6
shape-:+???????????????????????????
?)
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_text1serving_default_input_text2serving_default_input_text3"gcn_attention_30/conv1d_240/kernel gcn_attention_30/conv1d_240/bias4gcn_attention_30/batch_normalization_480/moving_mean8gcn_attention_30/batch_normalization_480/moving_variance-gcn_attention_30/batch_normalization_480/beta.gcn_attention_30/batch_normalization_480/gamma gcn_attention_30/dense_30/kernel"gcn_attention_30/conv2d_121/kernel gcn_attention_30/conv2d_121/bias.gcn_attention_30/batch_normalization_482/gamma-gcn_attention_30/batch_normalization_482/beta4gcn_attention_30/batch_normalization_482/moving_mean8gcn_attention_30/batch_normalization_482/moving_variance(gcn_attention_30/embedding_30/embeddings.gcn_attention_30/core_gcn_60/conv2d_122/kernel,gcn_attention_30/core_gcn_60/conv2d_122/bias:gcn_attention_30/core_gcn_60/batch_normalization_484/gamma9gcn_attention_30/core_gcn_60/batch_normalization_484/beta@gcn_attention_30/core_gcn_60/batch_normalization_484/moving_meanDgcn_attention_30/core_gcn_60/batch_normalization_484/moving_variance.gcn_attention_30/core_gcn_60/conv1d_241/kernel,gcn_attention_30/core_gcn_60/conv1d_241/bias@gcn_attention_30/core_gcn_60/batch_normalization_485/moving_meanDgcn_attention_30/core_gcn_60/batch_normalization_485/moving_variance9gcn_attention_30/core_gcn_60/batch_normalization_485/beta:gcn_attention_30/core_gcn_60/batch_normalization_485/gamma.gcn_attention_30/core_gcn_60/conv1d_242/kernel,gcn_attention_30/core_gcn_60/conv1d_242/bias@gcn_attention_30/core_gcn_60/batch_normalization_486/moving_meanDgcn_attention_30/core_gcn_60/batch_normalization_486/moving_variance9gcn_attention_30/core_gcn_60/batch_normalization_486/beta:gcn_attention_30/core_gcn_60/batch_normalization_486/gamma.gcn_attention_30/core_gcn_60/conv1d_243/kernel,gcn_attention_30/core_gcn_60/conv1d_243/bias@gcn_attention_30/core_gcn_60/batch_normalization_487/moving_meanDgcn_attention_30/core_gcn_60/batch_normalization_487/moving_variance9gcn_attention_30/core_gcn_60/batch_normalization_487/beta:gcn_attention_30/core_gcn_60/batch_normalization_487/gamma:gcn_attention_30/core_gcn_60/batch_normalization_488/gamma9gcn_attention_30/core_gcn_60/batch_normalization_488/beta@gcn_attention_30/core_gcn_60/batch_normalization_488/moving_meanDgcn_attention_30/core_gcn_60/batch_normalization_488/moving_variance@gcn_attention_30/core_gcn_60/batch_normalization_489/moving_meanDgcn_attention_30/core_gcn_60/batch_normalization_489/moving_variance9gcn_attention_30/core_gcn_60/batch_normalization_489/beta:gcn_attention_30/core_gcn_60/batch_normalization_489/gamma.gcn_attention_30/core_gcn_61/conv2d_123/kernel,gcn_attention_30/core_gcn_61/conv2d_123/bias:gcn_attention_30/core_gcn_61/batch_normalization_490/gamma9gcn_attention_30/core_gcn_61/batch_normalization_490/beta@gcn_attention_30/core_gcn_61/batch_normalization_490/moving_meanDgcn_attention_30/core_gcn_61/batch_normalization_490/moving_variance.gcn_attention_30/core_gcn_61/conv1d_244/kernel,gcn_attention_30/core_gcn_61/conv1d_244/bias@gcn_attention_30/core_gcn_61/batch_normalization_491/moving_meanDgcn_attention_30/core_gcn_61/batch_normalization_491/moving_variance9gcn_attention_30/core_gcn_61/batch_normalization_491/beta:gcn_attention_30/core_gcn_61/batch_normalization_491/gamma.gcn_attention_30/core_gcn_61/conv1d_245/kernel,gcn_attention_30/core_gcn_61/conv1d_245/bias@gcn_attention_30/core_gcn_61/batch_normalization_492/moving_meanDgcn_attention_30/core_gcn_61/batch_normalization_492/moving_variance9gcn_attention_30/core_gcn_61/batch_normalization_492/beta:gcn_attention_30/core_gcn_61/batch_normalization_492/gamma.gcn_attention_30/core_gcn_61/conv1d_246/kernel,gcn_attention_30/core_gcn_61/conv1d_246/bias@gcn_attention_30/core_gcn_61/batch_normalization_493/moving_meanDgcn_attention_30/core_gcn_61/batch_normalization_493/moving_variance9gcn_attention_30/core_gcn_61/batch_normalization_493/beta:gcn_attention_30/core_gcn_61/batch_normalization_493/gamma:gcn_attention_30/core_gcn_61/batch_normalization_494/gamma9gcn_attention_30/core_gcn_61/batch_normalization_494/beta@gcn_attention_30/core_gcn_61/batch_normalization_494/moving_meanDgcn_attention_30/core_gcn_61/batch_normalization_494/moving_variance@gcn_attention_30/core_gcn_61/batch_normalization_495/moving_meanDgcn_attention_30/core_gcn_61/batch_normalization_495/moving_variance9gcn_attention_30/core_gcn_61/batch_normalization_495/beta:gcn_attention_30/core_gcn_61/batch_normalization_495/gamma4gcn_attention_30/batch_normalization_483/moving_mean8gcn_attention_30/batch_normalization_483/moving_variance-gcn_attention_30/batch_normalization_483/beta.gcn_attention_30/batch_normalization_483/gamma"gcn_attention_30/conv1d_247/kernel gcn_attention_30/conv1d_247/bias*b
Tin[
Y2W*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*v
_read_only_resource_inputsX
VT	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUV*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_377124
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?4
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename6gcn_attention_30/conv1d_240/kernel/Read/ReadVariableOp4gcn_attention_30/conv1d_240/bias/Read/ReadVariableOpBgcn_attention_30/batch_normalization_480/gamma/Read/ReadVariableOpAgcn_attention_30/batch_normalization_480/beta/Read/ReadVariableOpHgcn_attention_30/batch_normalization_480/moving_mean/Read/ReadVariableOpLgcn_attention_30/batch_normalization_480/moving_variance/Read/ReadVariableOp4gcn_attention_30/dense_30/kernel/Read/ReadVariableOp6gcn_attention_30/conv2d_121/kernel/Read/ReadVariableOp4gcn_attention_30/conv2d_121/bias/Read/ReadVariableOpBgcn_attention_30/batch_normalization_482/gamma/Read/ReadVariableOpAgcn_attention_30/batch_normalization_482/beta/Read/ReadVariableOpHgcn_attention_30/batch_normalization_482/moving_mean/Read/ReadVariableOpLgcn_attention_30/batch_normalization_482/moving_variance/Read/ReadVariableOp<gcn_attention_30/embedding_30/embeddings/Read/ReadVariableOpBgcn_attention_30/batch_normalization_483/gamma/Read/ReadVariableOpAgcn_attention_30/batch_normalization_483/beta/Read/ReadVariableOpHgcn_attention_30/batch_normalization_483/moving_mean/Read/ReadVariableOpLgcn_attention_30/batch_normalization_483/moving_variance/Read/ReadVariableOp6gcn_attention_30/conv1d_247/kernel/Read/ReadVariableOp4gcn_attention_30/conv1d_247/bias/Read/ReadVariableOpBgcn_attention_30/core_gcn_60/conv2d_122/kernel/Read/ReadVariableOp@gcn_attention_30/core_gcn_60/conv2d_122/bias/Read/ReadVariableOpNgcn_attention_30/core_gcn_60/batch_normalization_484/gamma/Read/ReadVariableOpMgcn_attention_30/core_gcn_60/batch_normalization_484/beta/Read/ReadVariableOpBgcn_attention_30/core_gcn_60/conv1d_241/kernel/Read/ReadVariableOp@gcn_attention_30/core_gcn_60/conv1d_241/bias/Read/ReadVariableOpNgcn_attention_30/core_gcn_60/batch_normalization_485/gamma/Read/ReadVariableOpMgcn_attention_30/core_gcn_60/batch_normalization_485/beta/Read/ReadVariableOpBgcn_attention_30/core_gcn_60/conv1d_242/kernel/Read/ReadVariableOp@gcn_attention_30/core_gcn_60/conv1d_242/bias/Read/ReadVariableOpNgcn_attention_30/core_gcn_60/batch_normalization_486/gamma/Read/ReadVariableOpMgcn_attention_30/core_gcn_60/batch_normalization_486/beta/Read/ReadVariableOpBgcn_attention_30/core_gcn_60/conv1d_243/kernel/Read/ReadVariableOp@gcn_attention_30/core_gcn_60/conv1d_243/bias/Read/ReadVariableOpNgcn_attention_30/core_gcn_60/batch_normalization_487/gamma/Read/ReadVariableOpMgcn_attention_30/core_gcn_60/batch_normalization_487/beta/Read/ReadVariableOpNgcn_attention_30/core_gcn_60/batch_normalization_488/gamma/Read/ReadVariableOpMgcn_attention_30/core_gcn_60/batch_normalization_488/beta/Read/ReadVariableOpNgcn_attention_30/core_gcn_60/batch_normalization_489/gamma/Read/ReadVariableOpMgcn_attention_30/core_gcn_60/batch_normalization_489/beta/Read/ReadVariableOpBgcn_attention_30/core_gcn_61/conv2d_123/kernel/Read/ReadVariableOp@gcn_attention_30/core_gcn_61/conv2d_123/bias/Read/ReadVariableOpNgcn_attention_30/core_gcn_61/batch_normalization_490/gamma/Read/ReadVariableOpMgcn_attention_30/core_gcn_61/batch_normalization_490/beta/Read/ReadVariableOpBgcn_attention_30/core_gcn_61/conv1d_244/kernel/Read/ReadVariableOp@gcn_attention_30/core_gcn_61/conv1d_244/bias/Read/ReadVariableOpNgcn_attention_30/core_gcn_61/batch_normalization_491/gamma/Read/ReadVariableOpMgcn_attention_30/core_gcn_61/batch_normalization_491/beta/Read/ReadVariableOpBgcn_attention_30/core_gcn_61/conv1d_245/kernel/Read/ReadVariableOp@gcn_attention_30/core_gcn_61/conv1d_245/bias/Read/ReadVariableOpNgcn_attention_30/core_gcn_61/batch_normalization_492/gamma/Read/ReadVariableOpMgcn_attention_30/core_gcn_61/batch_normalization_492/beta/Read/ReadVariableOpBgcn_attention_30/core_gcn_61/conv1d_246/kernel/Read/ReadVariableOp@gcn_attention_30/core_gcn_61/conv1d_246/bias/Read/ReadVariableOpNgcn_attention_30/core_gcn_61/batch_normalization_493/gamma/Read/ReadVariableOpMgcn_attention_30/core_gcn_61/batch_normalization_493/beta/Read/ReadVariableOpNgcn_attention_30/core_gcn_61/batch_normalization_494/gamma/Read/ReadVariableOpMgcn_attention_30/core_gcn_61/batch_normalization_494/beta/Read/ReadVariableOpNgcn_attention_30/core_gcn_61/batch_normalization_495/gamma/Read/ReadVariableOpMgcn_attention_30/core_gcn_61/batch_normalization_495/beta/Read/ReadVariableOpTgcn_attention_30/core_gcn_60/batch_normalization_484/moving_mean/Read/ReadVariableOpXgcn_attention_30/core_gcn_60/batch_normalization_484/moving_variance/Read/ReadVariableOpTgcn_attention_30/core_gcn_60/batch_normalization_485/moving_mean/Read/ReadVariableOpXgcn_attention_30/core_gcn_60/batch_normalization_485/moving_variance/Read/ReadVariableOpTgcn_attention_30/core_gcn_60/batch_normalization_486/moving_mean/Read/ReadVariableOpXgcn_attention_30/core_gcn_60/batch_normalization_486/moving_variance/Read/ReadVariableOpTgcn_attention_30/core_gcn_60/batch_normalization_487/moving_mean/Read/ReadVariableOpXgcn_attention_30/core_gcn_60/batch_normalization_487/moving_variance/Read/ReadVariableOpTgcn_attention_30/core_gcn_60/batch_normalization_488/moving_mean/Read/ReadVariableOpXgcn_attention_30/core_gcn_60/batch_normalization_488/moving_variance/Read/ReadVariableOpTgcn_attention_30/core_gcn_60/batch_normalization_489/moving_mean/Read/ReadVariableOpXgcn_attention_30/core_gcn_60/batch_normalization_489/moving_variance/Read/ReadVariableOpTgcn_attention_30/core_gcn_61/batch_normalization_490/moving_mean/Read/ReadVariableOpXgcn_attention_30/core_gcn_61/batch_normalization_490/moving_variance/Read/ReadVariableOpTgcn_attention_30/core_gcn_61/batch_normalization_491/moving_mean/Read/ReadVariableOpXgcn_attention_30/core_gcn_61/batch_normalization_491/moving_variance/Read/ReadVariableOpTgcn_attention_30/core_gcn_61/batch_normalization_492/moving_mean/Read/ReadVariableOpXgcn_attention_30/core_gcn_61/batch_normalization_492/moving_variance/Read/ReadVariableOpTgcn_attention_30/core_gcn_61/batch_normalization_493/moving_mean/Read/ReadVariableOpXgcn_attention_30/core_gcn_61/batch_normalization_493/moving_variance/Read/ReadVariableOpTgcn_attention_30/core_gcn_61/batch_normalization_494/moving_mean/Read/ReadVariableOpXgcn_attention_30/core_gcn_61/batch_normalization_494/moving_variance/Read/ReadVariableOpTgcn_attention_30/core_gcn_61/batch_normalization_495/moving_mean/Read/ReadVariableOpXgcn_attention_30/core_gcn_61/batch_normalization_495/moving_variance/Read/ReadVariableOpConst*a
TinZ
X2V*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_380865
?'
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename"gcn_attention_30/conv1d_240/kernel gcn_attention_30/conv1d_240/bias.gcn_attention_30/batch_normalization_480/gamma-gcn_attention_30/batch_normalization_480/beta4gcn_attention_30/batch_normalization_480/moving_mean8gcn_attention_30/batch_normalization_480/moving_variance gcn_attention_30/dense_30/kernel"gcn_attention_30/conv2d_121/kernel gcn_attention_30/conv2d_121/bias.gcn_attention_30/batch_normalization_482/gamma-gcn_attention_30/batch_normalization_482/beta4gcn_attention_30/batch_normalization_482/moving_mean8gcn_attention_30/batch_normalization_482/moving_variance(gcn_attention_30/embedding_30/embeddings.gcn_attention_30/batch_normalization_483/gamma-gcn_attention_30/batch_normalization_483/beta4gcn_attention_30/batch_normalization_483/moving_mean8gcn_attention_30/batch_normalization_483/moving_variance"gcn_attention_30/conv1d_247/kernel gcn_attention_30/conv1d_247/bias.gcn_attention_30/core_gcn_60/conv2d_122/kernel,gcn_attention_30/core_gcn_60/conv2d_122/bias:gcn_attention_30/core_gcn_60/batch_normalization_484/gamma9gcn_attention_30/core_gcn_60/batch_normalization_484/beta.gcn_attention_30/core_gcn_60/conv1d_241/kernel,gcn_attention_30/core_gcn_60/conv1d_241/bias:gcn_attention_30/core_gcn_60/batch_normalization_485/gamma9gcn_attention_30/core_gcn_60/batch_normalization_485/beta.gcn_attention_30/core_gcn_60/conv1d_242/kernel,gcn_attention_30/core_gcn_60/conv1d_242/bias:gcn_attention_30/core_gcn_60/batch_normalization_486/gamma9gcn_attention_30/core_gcn_60/batch_normalization_486/beta.gcn_attention_30/core_gcn_60/conv1d_243/kernel,gcn_attention_30/core_gcn_60/conv1d_243/bias:gcn_attention_30/core_gcn_60/batch_normalization_487/gamma9gcn_attention_30/core_gcn_60/batch_normalization_487/beta:gcn_attention_30/core_gcn_60/batch_normalization_488/gamma9gcn_attention_30/core_gcn_60/batch_normalization_488/beta:gcn_attention_30/core_gcn_60/batch_normalization_489/gamma9gcn_attention_30/core_gcn_60/batch_normalization_489/beta.gcn_attention_30/core_gcn_61/conv2d_123/kernel,gcn_attention_30/core_gcn_61/conv2d_123/bias:gcn_attention_30/core_gcn_61/batch_normalization_490/gamma9gcn_attention_30/core_gcn_61/batch_normalization_490/beta.gcn_attention_30/core_gcn_61/conv1d_244/kernel,gcn_attention_30/core_gcn_61/conv1d_244/bias:gcn_attention_30/core_gcn_61/batch_normalization_491/gamma9gcn_attention_30/core_gcn_61/batch_normalization_491/beta.gcn_attention_30/core_gcn_61/conv1d_245/kernel,gcn_attention_30/core_gcn_61/conv1d_245/bias:gcn_attention_30/core_gcn_61/batch_normalization_492/gamma9gcn_attention_30/core_gcn_61/batch_normalization_492/beta.gcn_attention_30/core_gcn_61/conv1d_246/kernel,gcn_attention_30/core_gcn_61/conv1d_246/bias:gcn_attention_30/core_gcn_61/batch_normalization_493/gamma9gcn_attention_30/core_gcn_61/batch_normalization_493/beta:gcn_attention_30/core_gcn_61/batch_normalization_494/gamma9gcn_attention_30/core_gcn_61/batch_normalization_494/beta:gcn_attention_30/core_gcn_61/batch_normalization_495/gamma9gcn_attention_30/core_gcn_61/batch_normalization_495/beta@gcn_attention_30/core_gcn_60/batch_normalization_484/moving_meanDgcn_attention_30/core_gcn_60/batch_normalization_484/moving_variance@gcn_attention_30/core_gcn_60/batch_normalization_485/moving_meanDgcn_attention_30/core_gcn_60/batch_normalization_485/moving_variance@gcn_attention_30/core_gcn_60/batch_normalization_486/moving_meanDgcn_attention_30/core_gcn_60/batch_normalization_486/moving_variance@gcn_attention_30/core_gcn_60/batch_normalization_487/moving_meanDgcn_attention_30/core_gcn_60/batch_normalization_487/moving_variance@gcn_attention_30/core_gcn_60/batch_normalization_488/moving_meanDgcn_attention_30/core_gcn_60/batch_normalization_488/moving_variance@gcn_attention_30/core_gcn_60/batch_normalization_489/moving_meanDgcn_attention_30/core_gcn_60/batch_normalization_489/moving_variance@gcn_attention_30/core_gcn_61/batch_normalization_490/moving_meanDgcn_attention_30/core_gcn_61/batch_normalization_490/moving_variance@gcn_attention_30/core_gcn_61/batch_normalization_491/moving_meanDgcn_attention_30/core_gcn_61/batch_normalization_491/moving_variance@gcn_attention_30/core_gcn_61/batch_normalization_492/moving_meanDgcn_attention_30/core_gcn_61/batch_normalization_492/moving_variance@gcn_attention_30/core_gcn_61/batch_normalization_493/moving_meanDgcn_attention_30/core_gcn_61/batch_normalization_493/moving_variance@gcn_attention_30/core_gcn_61/batch_normalization_494/moving_meanDgcn_attention_30/core_gcn_61/batch_normalization_494/moving_variance@gcn_attention_30/core_gcn_61/batch_normalization_495/moving_meanDgcn_attention_30/core_gcn_61/batch_normalization_495/moving_variance*`
TinY
W2U*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_381127??"
?	
?
8__inference_batch_normalization_487_layer_call_fn_379907

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_487_layer_call_and_return_conditional_losses_3780482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_485_layer_call_fn_379760

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_485_layer_call_and_return_conditional_losses_3777842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?*
?
S__inference_batch_normalization_491_layer_call_and_return_conditional_losses_378684

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_486_layer_call_and_return_conditional_losses_377886

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_486_layer_call_fn_379827

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_486_layer_call_and_return_conditional_losses_3778862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?*
?
S__inference_batch_normalization_489_layer_call_and_return_conditional_losses_380116

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_493_layer_call_and_return_conditional_losses_378948

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_488_layer_call_fn_379987

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_488_layer_call_and_return_conditional_losses_3782082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?*
?
S__inference_batch_normalization_492_layer_call_and_return_conditional_losses_378846

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_494_layer_call_fn_380461

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_494_layer_call_and_return_conditional_losses_3791522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_495_layer_call_fn_380510

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_495_layer_call_and_return_conditional_losses_3792362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_489_layer_call_and_return_conditional_losses_380082

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?*
?
S__inference_batch_normalization_486_layer_call_and_return_conditional_losses_377946

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_380127s
Ygcn_attention_30_core_gcn_60_conv2d_122_kernel_regularizer_square_readvariableop_resource:
identity??Pgcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/Square/ReadVariableOp?
Pgcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/Square/ReadVariableOpReadVariableOpYgcn_attention_30_core_gcn_60_conv2d_122_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype02R
Pgcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/Square/ReadVariableOp?
Agcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/SquareSquareXgcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2C
Agcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/Square?
@gcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2B
@gcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/Const?
>gcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/SumSumEgcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/Square:y:0Igcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2@
>gcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/Sum?
@gcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82B
@gcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/mul/x?
>gcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/mulMulIgcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/mul/x:output:0Ggcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2@
>gcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/mul?
IdentityIdentityBgcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOpQ^gcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2?
Pgcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/Square/ReadVariableOpPgcn_attention_30/core_gcn_60/conv2d_122/kernel/Regularizer/Square/ReadVariableOp
?
?
S__inference_batch_normalization_491_layer_call_and_return_conditional_losses_380241

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_488_layer_call_and_return_conditional_losses_380018

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?*
?
S__inference_batch_normalization_486_layer_call_and_return_conditional_losses_379894

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_484_layer_call_and_return_conditional_losses_379734

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?*
?
S__inference_batch_normalization_483_layer_call_and_return_conditional_losses_377496

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_488_layer_call_and_return_conditional_losses_380036

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?:
__inference__traced_save_380865
file_prefixA
=savev2_gcn_attention_30_conv1d_240_kernel_read_readvariableop?
;savev2_gcn_attention_30_conv1d_240_bias_read_readvariableopM
Isavev2_gcn_attention_30_batch_normalization_480_gamma_read_readvariableopL
Hsavev2_gcn_attention_30_batch_normalization_480_beta_read_readvariableopS
Osavev2_gcn_attention_30_batch_normalization_480_moving_mean_read_readvariableopW
Ssavev2_gcn_attention_30_batch_normalization_480_moving_variance_read_readvariableop?
;savev2_gcn_attention_30_dense_30_kernel_read_readvariableopA
=savev2_gcn_attention_30_conv2d_121_kernel_read_readvariableop?
;savev2_gcn_attention_30_conv2d_121_bias_read_readvariableopM
Isavev2_gcn_attention_30_batch_normalization_482_gamma_read_readvariableopL
Hsavev2_gcn_attention_30_batch_normalization_482_beta_read_readvariableopS
Osavev2_gcn_attention_30_batch_normalization_482_moving_mean_read_readvariableopW
Ssavev2_gcn_attention_30_batch_normalization_482_moving_variance_read_readvariableopG
Csavev2_gcn_attention_30_embedding_30_embeddings_read_readvariableopM
Isavev2_gcn_attention_30_batch_normalization_483_gamma_read_readvariableopL
Hsavev2_gcn_attention_30_batch_normalization_483_beta_read_readvariableopS
Osavev2_gcn_attention_30_batch_normalization_483_moving_mean_read_readvariableopW
Ssavev2_gcn_attention_30_batch_normalization_483_moving_variance_read_readvariableopA
=savev2_gcn_attention_30_conv1d_247_kernel_read_readvariableop?
;savev2_gcn_attention_30_conv1d_247_bias_read_readvariableopM
Isavev2_gcn_attention_30_core_gcn_60_conv2d_122_kernel_read_readvariableopK
Gsavev2_gcn_attention_30_core_gcn_60_conv2d_122_bias_read_readvariableopY
Usavev2_gcn_attention_30_core_gcn_60_batch_normalization_484_gamma_read_readvariableopX
Tsavev2_gcn_attention_30_core_gcn_60_batch_normalization_484_beta_read_readvariableopM
Isavev2_gcn_attention_30_core_gcn_60_conv1d_241_kernel_read_readvariableopK
Gsavev2_gcn_attention_30_core_gcn_60_conv1d_241_bias_read_readvariableopY
Usavev2_gcn_attention_30_core_gcn_60_batch_normalization_485_gamma_read_readvariableopX
Tsavev2_gcn_attention_30_core_gcn_60_batch_normalization_485_beta_read_readvariableopM
Isavev2_gcn_attention_30_core_gcn_60_conv1d_242_kernel_read_readvariableopK
Gsavev2_gcn_attention_30_core_gcn_60_conv1d_242_bias_read_readvariableopY
Usavev2_gcn_attention_30_core_gcn_60_batch_normalization_486_gamma_read_readvariableopX
Tsavev2_gcn_attention_30_core_gcn_60_batch_normalization_486_beta_read_readvariableopM
Isavev2_gcn_attention_30_core_gcn_60_conv1d_243_kernel_read_readvariableopK
Gsavev2_gcn_attention_30_core_gcn_60_conv1d_243_bias_read_readvariableopY
Usavev2_gcn_attention_30_core_gcn_60_batch_normalization_487_gamma_read_readvariableopX
Tsavev2_gcn_attention_30_core_gcn_60_batch_normalization_487_beta_read_readvariableopY
Usavev2_gcn_attention_30_core_gcn_60_batch_normalization_488_gamma_read_readvariableopX
Tsavev2_gcn_attention_30_core_gcn_60_batch_normalization_488_beta_read_readvariableopY
Usavev2_gcn_attention_30_core_gcn_60_batch_normalization_489_gamma_read_readvariableopX
Tsavev2_gcn_attention_30_core_gcn_60_batch_normalization_489_beta_read_readvariableopM
Isavev2_gcn_attention_30_core_gcn_61_conv2d_123_kernel_read_readvariableopK
Gsavev2_gcn_attention_30_core_gcn_61_conv2d_123_bias_read_readvariableopY
Usavev2_gcn_attention_30_core_gcn_61_batch_normalization_490_gamma_read_readvariableopX
Tsavev2_gcn_attention_30_core_gcn_61_batch_normalization_490_beta_read_readvariableopM
Isavev2_gcn_attention_30_core_gcn_61_conv1d_244_kernel_read_readvariableopK
Gsavev2_gcn_attention_30_core_gcn_61_conv1d_244_bias_read_readvariableopY
Usavev2_gcn_attention_30_core_gcn_61_batch_normalization_491_gamma_read_readvariableopX
Tsavev2_gcn_attention_30_core_gcn_61_batch_normalization_491_beta_read_readvariableopM
Isavev2_gcn_attention_30_core_gcn_61_conv1d_245_kernel_read_readvariableopK
Gsavev2_gcn_attention_30_core_gcn_61_conv1d_245_bias_read_readvariableopY
Usavev2_gcn_attention_30_core_gcn_61_batch_normalization_492_gamma_read_readvariableopX
Tsavev2_gcn_attention_30_core_gcn_61_batch_normalization_492_beta_read_readvariableopM
Isavev2_gcn_attention_30_core_gcn_61_conv1d_246_kernel_read_readvariableopK
Gsavev2_gcn_attention_30_core_gcn_61_conv1d_246_bias_read_readvariableopY
Usavev2_gcn_attention_30_core_gcn_61_batch_normalization_493_gamma_read_readvariableopX
Tsavev2_gcn_attention_30_core_gcn_61_batch_normalization_493_beta_read_readvariableopY
Usavev2_gcn_attention_30_core_gcn_61_batch_normalization_494_gamma_read_readvariableopX
Tsavev2_gcn_attention_30_core_gcn_61_batch_normalization_494_beta_read_readvariableopY
Usavev2_gcn_attention_30_core_gcn_61_batch_normalization_495_gamma_read_readvariableopX
Tsavev2_gcn_attention_30_core_gcn_61_batch_normalization_495_beta_read_readvariableop_
[savev2_gcn_attention_30_core_gcn_60_batch_normalization_484_moving_mean_read_readvariableopc
_savev2_gcn_attention_30_core_gcn_60_batch_normalization_484_moving_variance_read_readvariableop_
[savev2_gcn_attention_30_core_gcn_60_batch_normalization_485_moving_mean_read_readvariableopc
_savev2_gcn_attention_30_core_gcn_60_batch_normalization_485_moving_variance_read_readvariableop_
[savev2_gcn_attention_30_core_gcn_60_batch_normalization_486_moving_mean_read_readvariableopc
_savev2_gcn_attention_30_core_gcn_60_batch_normalization_486_moving_variance_read_readvariableop_
[savev2_gcn_attention_30_core_gcn_60_batch_normalization_487_moving_mean_read_readvariableopc
_savev2_gcn_attention_30_core_gcn_60_batch_normalization_487_moving_variance_read_readvariableop_
[savev2_gcn_attention_30_core_gcn_60_batch_normalization_488_moving_mean_read_readvariableopc
_savev2_gcn_attention_30_core_gcn_60_batch_normalization_488_moving_variance_read_readvariableop_
[savev2_gcn_attention_30_core_gcn_60_batch_normalization_489_moving_mean_read_readvariableopc
_savev2_gcn_attention_30_core_gcn_60_batch_normalization_489_moving_variance_read_readvariableop_
[savev2_gcn_attention_30_core_gcn_61_batch_normalization_490_moving_mean_read_readvariableopc
_savev2_gcn_attention_30_core_gcn_61_batch_normalization_490_moving_variance_read_readvariableop_
[savev2_gcn_attention_30_core_gcn_61_batch_normalization_491_moving_mean_read_readvariableopc
_savev2_gcn_attention_30_core_gcn_61_batch_normalization_491_moving_variance_read_readvariableop_
[savev2_gcn_attention_30_core_gcn_61_batch_normalization_492_moving_mean_read_readvariableopc
_savev2_gcn_attention_30_core_gcn_61_batch_normalization_492_moving_variance_read_readvariableop_
[savev2_gcn_attention_30_core_gcn_61_batch_normalization_493_moving_mean_read_readvariableopc
_savev2_gcn_attention_30_core_gcn_61_batch_normalization_493_moving_variance_read_readvariableop_
[savev2_gcn_attention_30_core_gcn_61_batch_normalization_494_moving_mean_read_readvariableopc
_savev2_gcn_attention_30_core_gcn_61_batch_normalization_494_moving_variance_read_readvariableop_
[savev2_gcn_attention_30_core_gcn_61_batch_normalization_495_moving_mean_read_readvariableopc
_savev2_gcn_attention_30_core_gcn_61_batch_normalization_495_moving_variance_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?,
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:U*
dtype0*?+
value?+B?+UBCself_attention_layer/node0_Conv1D/kernel/.ATTRIBUTES/VARIABLE_VALUEBAself_attention_layer/node0_Conv1D/bias/.ATTRIBUTES/VARIABLE_VALUEBNself_attention_layer/node0_BatchNormalization/gamma/.ATTRIBUTES/VARIABLE_VALUEBMself_attention_layer/node0_BatchNormalization/beta/.ATTRIBUTES/VARIABLE_VALUEBTself_attention_layer/node0_BatchNormalization/moving_mean/.ATTRIBUTES/VARIABLE_VALUEBXself_attention_layer/node0_BatchNormalization/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBEself_attention_layer/distance_Dense/kernel/.ATTRIBUTES/VARIABLE_VALUEBMself_attention_layer/distance_Convolution2D/kernel/.ATTRIBUTES/VARIABLE_VALUEBKself_attention_layer/distance_Convolution2D/bias/.ATTRIBUTES/VARIABLE_VALUEBQself_attention_layer/distance_BatchNormalization/gamma/.ATTRIBUTES/VARIABLE_VALUEBPself_attention_layer/distance_BatchNormalization/beta/.ATTRIBUTES/VARIABLE_VALUEBWself_attention_layer/distance_BatchNormalization/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB[self_attention_layer/distance_BatchNormalization/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBDself_attention_layer/adj_Dense/embeddings/.ATTRIBUTES/VARIABLE_VALUEBNself_attention_layer/node_BatchNormalization5/gamma/.ATTRIBUTES/VARIABLE_VALUEBMself_attention_layer/node_BatchNormalization5/beta/.ATTRIBUTES/VARIABLE_VALUEBTself_attention_layer/node_BatchNormalization5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEBXself_attention_layer/node_BatchNormalization5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/node_Conv1D_out/kernel/.ATTRIBUTES/VARIABLE_VALUEBDself_attention_layer/node_Conv1D_out/bias/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/12/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/13/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/14/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/15/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/16/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/17/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/18/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/19/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/20/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/21/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/22/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/23/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/24/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/25/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/26/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/27/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/28/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/29/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/30/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/31/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/32/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/33/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/34/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/35/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/36/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/37/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/38/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/39/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/40/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/41/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/42/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/43/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/44/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/45/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/46/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/47/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/48/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/49/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/50/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/51/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/60/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/61/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/62/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/63/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/64/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/65/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/66/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/67/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/68/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/69/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/70/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/71/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/72/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/73/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/74/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/75/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/76/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/77/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/78/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/79/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/80/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/81/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/82/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/83/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:U*
dtype0*?
value?B?UB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?8
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0=savev2_gcn_attention_30_conv1d_240_kernel_read_readvariableop;savev2_gcn_attention_30_conv1d_240_bias_read_readvariableopIsavev2_gcn_attention_30_batch_normalization_480_gamma_read_readvariableopHsavev2_gcn_attention_30_batch_normalization_480_beta_read_readvariableopOsavev2_gcn_attention_30_batch_normalization_480_moving_mean_read_readvariableopSsavev2_gcn_attention_30_batch_normalization_480_moving_variance_read_readvariableop;savev2_gcn_attention_30_dense_30_kernel_read_readvariableop=savev2_gcn_attention_30_conv2d_121_kernel_read_readvariableop;savev2_gcn_attention_30_conv2d_121_bias_read_readvariableopIsavev2_gcn_attention_30_batch_normalization_482_gamma_read_readvariableopHsavev2_gcn_attention_30_batch_normalization_482_beta_read_readvariableopOsavev2_gcn_attention_30_batch_normalization_482_moving_mean_read_readvariableopSsavev2_gcn_attention_30_batch_normalization_482_moving_variance_read_readvariableopCsavev2_gcn_attention_30_embedding_30_embeddings_read_readvariableopIsavev2_gcn_attention_30_batch_normalization_483_gamma_read_readvariableopHsavev2_gcn_attention_30_batch_normalization_483_beta_read_readvariableopOsavev2_gcn_attention_30_batch_normalization_483_moving_mean_read_readvariableopSsavev2_gcn_attention_30_batch_normalization_483_moving_variance_read_readvariableop=savev2_gcn_attention_30_conv1d_247_kernel_read_readvariableop;savev2_gcn_attention_30_conv1d_247_bias_read_readvariableopIsavev2_gcn_attention_30_core_gcn_60_conv2d_122_kernel_read_readvariableopGsavev2_gcn_attention_30_core_gcn_60_conv2d_122_bias_read_readvariableopUsavev2_gcn_attention_30_core_gcn_60_batch_normalization_484_gamma_read_readvariableopTsavev2_gcn_attention_30_core_gcn_60_batch_normalization_484_beta_read_readvariableopIsavev2_gcn_attention_30_core_gcn_60_conv1d_241_kernel_read_readvariableopGsavev2_gcn_attention_30_core_gcn_60_conv1d_241_bias_read_readvariableopUsavev2_gcn_attention_30_core_gcn_60_batch_normalization_485_gamma_read_readvariableopTsavev2_gcn_attention_30_core_gcn_60_batch_normalization_485_beta_read_readvariableopIsavev2_gcn_attention_30_core_gcn_60_conv1d_242_kernel_read_readvariableopGsavev2_gcn_attention_30_core_gcn_60_conv1d_242_bias_read_readvariableopUsavev2_gcn_attention_30_core_gcn_60_batch_normalization_486_gamma_read_readvariableopTsavev2_gcn_attention_30_core_gcn_60_batch_normalization_486_beta_read_readvariableopIsavev2_gcn_attention_30_core_gcn_60_conv1d_243_kernel_read_readvariableopGsavev2_gcn_attention_30_core_gcn_60_conv1d_243_bias_read_readvariableopUsavev2_gcn_attention_30_core_gcn_60_batch_normalization_487_gamma_read_readvariableopTsavev2_gcn_attention_30_core_gcn_60_batch_normalization_487_beta_read_readvariableopUsavev2_gcn_attention_30_core_gcn_60_batch_normalization_488_gamma_read_readvariableopTsavev2_gcn_attention_30_core_gcn_60_batch_normalization_488_beta_read_readvariableopUsavev2_gcn_attention_30_core_gcn_60_batch_normalization_489_gamma_read_readvariableopTsavev2_gcn_attention_30_core_gcn_60_batch_normalization_489_beta_read_readvariableopIsavev2_gcn_attention_30_core_gcn_61_conv2d_123_kernel_read_readvariableopGsavev2_gcn_attention_30_core_gcn_61_conv2d_123_bias_read_readvariableopUsavev2_gcn_attention_30_core_gcn_61_batch_normalization_490_gamma_read_readvariableopTsavev2_gcn_attention_30_core_gcn_61_batch_normalization_490_beta_read_readvariableopIsavev2_gcn_attention_30_core_gcn_61_conv1d_244_kernel_read_readvariableopGsavev2_gcn_attention_30_core_gcn_61_conv1d_244_bias_read_readvariableopUsavev2_gcn_attention_30_core_gcn_61_batch_normalization_491_gamma_read_readvariableopTsavev2_gcn_attention_30_core_gcn_61_batch_normalization_491_beta_read_readvariableopIsavev2_gcn_attention_30_core_gcn_61_conv1d_245_kernel_read_readvariableopGsavev2_gcn_attention_30_core_gcn_61_conv1d_245_bias_read_readvariableopUsavev2_gcn_attention_30_core_gcn_61_batch_normalization_492_gamma_read_readvariableopTsavev2_gcn_attention_30_core_gcn_61_batch_normalization_492_beta_read_readvariableopIsavev2_gcn_attention_30_core_gcn_61_conv1d_246_kernel_read_readvariableopGsavev2_gcn_attention_30_core_gcn_61_conv1d_246_bias_read_readvariableopUsavev2_gcn_attention_30_core_gcn_61_batch_normalization_493_gamma_read_readvariableopTsavev2_gcn_attention_30_core_gcn_61_batch_normalization_493_beta_read_readvariableopUsavev2_gcn_attention_30_core_gcn_61_batch_normalization_494_gamma_read_readvariableopTsavev2_gcn_attention_30_core_gcn_61_batch_normalization_494_beta_read_readvariableopUsavev2_gcn_attention_30_core_gcn_61_batch_normalization_495_gamma_read_readvariableopTsavev2_gcn_attention_30_core_gcn_61_batch_normalization_495_beta_read_readvariableop[savev2_gcn_attention_30_core_gcn_60_batch_normalization_484_moving_mean_read_readvariableop_savev2_gcn_attention_30_core_gcn_60_batch_normalization_484_moving_variance_read_readvariableop[savev2_gcn_attention_30_core_gcn_60_batch_normalization_485_moving_mean_read_readvariableop_savev2_gcn_attention_30_core_gcn_60_batch_normalization_485_moving_variance_read_readvariableop[savev2_gcn_attention_30_core_gcn_60_batch_normalization_486_moving_mean_read_readvariableop_savev2_gcn_attention_30_core_gcn_60_batch_normalization_486_moving_variance_read_readvariableop[savev2_gcn_attention_30_core_gcn_60_batch_normalization_487_moving_mean_read_readvariableop_savev2_gcn_attention_30_core_gcn_60_batch_normalization_487_moving_variance_read_readvariableop[savev2_gcn_attention_30_core_gcn_60_batch_normalization_488_moving_mean_read_readvariableop_savev2_gcn_attention_30_core_gcn_60_batch_normalization_488_moving_variance_read_readvariableop[savev2_gcn_attention_30_core_gcn_60_batch_normalization_489_moving_mean_read_readvariableop_savev2_gcn_attention_30_core_gcn_60_batch_normalization_489_moving_variance_read_readvariableop[savev2_gcn_attention_30_core_gcn_61_batch_normalization_490_moving_mean_read_readvariableop_savev2_gcn_attention_30_core_gcn_61_batch_normalization_490_moving_variance_read_readvariableop[savev2_gcn_attention_30_core_gcn_61_batch_normalization_491_moving_mean_read_readvariableop_savev2_gcn_attention_30_core_gcn_61_batch_normalization_491_moving_variance_read_readvariableop[savev2_gcn_attention_30_core_gcn_61_batch_normalization_492_moving_mean_read_readvariableop_savev2_gcn_attention_30_core_gcn_61_batch_normalization_492_moving_variance_read_readvariableop[savev2_gcn_attention_30_core_gcn_61_batch_normalization_493_moving_mean_read_readvariableop_savev2_gcn_attention_30_core_gcn_61_batch_normalization_493_moving_variance_read_readvariableop[savev2_gcn_attention_30_core_gcn_61_batch_normalization_494_moving_mean_read_readvariableop_savev2_gcn_attention_30_core_gcn_61_batch_normalization_494_moving_variance_read_readvariableop[savev2_gcn_attention_30_core_gcn_61_batch_normalization_495_moving_mean_read_readvariableop_savev2_gcn_attention_30_core_gcn_61_batch_normalization_495_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *c
dtypesY
W2U2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :?::::::	:		:	:	:	:	:	:	::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_output_shapes
:?: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:	:,(
&
_output_shapes
:		: 	

_output_shapes
:	: 


_output_shapes
:	: 

_output_shapes
:	: 

_output_shapes
:	: 

_output_shapes
:	:$ 

_output_shapes

:	: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
::(!$
"
_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
:: (

_output_shapes
::,)(
&
_output_shapes
:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
::(-$
"
_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
::(1$
"
_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
::(5$
"
_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
:: <

_output_shapes
:: =

_output_shapes
:: >

_output_shapes
:: ?

_output_shapes
:: @

_output_shapes
:: A

_output_shapes
:: B

_output_shapes
:: C

_output_shapes
:: D

_output_shapes
:: E

_output_shapes
:: F

_output_shapes
:: G

_output_shapes
:: H

_output_shapes
:: I

_output_shapes
:: J

_output_shapes
:: K

_output_shapes
:: L

_output_shapes
:: M

_output_shapes
:: N

_output_shapes
:: O

_output_shapes
:: P

_output_shapes
:: Q

_output_shapes
:: R

_output_shapes
:: S

_output_shapes
:: T

_output_shapes
::U

_output_shapes
: 
?	
?
8__inference_batch_normalization_483_layer_call_fn_379578

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_483_layer_call_and_return_conditional_losses_3774962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?*
?
S__inference_batch_normalization_483_layer_call_and_return_conditional_losses_379632

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?*
?
S__inference_batch_normalization_480_layer_call_and_return_conditional_losses_377208

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_480_layer_call_and_return_conditional_losses_379450

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_482_layer_call_and_return_conditional_losses_377352

inputs%
readvariableop_resource:	'
readvariableop_1_resource:	6
(fusedbatchnormv3_readvariableop_resource:	8
*fusedbatchnormv3_readvariableop_1_resource:	
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:	*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:	*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????	:	:	:	:	:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????	2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????	: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????	
 
_user_specified_nameinputs
?*
?
S__inference_batch_normalization_493_layer_call_and_return_conditional_losses_379008

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_494_layer_call_and_return_conditional_losses_380479

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_484_layer_call_fn_379685

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_484_layer_call_and_return_conditional_losses_3775962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_487_layer_call_and_return_conditional_losses_379940

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_488_layer_call_and_return_conditional_losses_378208

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_483_layer_call_and_return_conditional_losses_377436

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?*
?
S__inference_batch_normalization_492_layer_call_and_return_conditional_losses_380355

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_490_layer_call_and_return_conditional_losses_380195

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_482_layer_call_and_return_conditional_losses_379534

inputs%
readvariableop_resource:	'
readvariableop_1_resource:	6
(fusedbatchnormv3_readvariableop_resource:	8
*fusedbatchnormv3_readvariableop_1_resource:	
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:	*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:	*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????	:	:	:	:	:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????	2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????	: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????	
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_493_layer_call_fn_380368

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_493_layer_call_and_return_conditional_losses_3789482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_494_layer_call_fn_380448

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_494_layer_call_and_return_conditional_losses_3791082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_495_layer_call_fn_380523

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_495_layer_call_and_return_conditional_losses_3792962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_484_layer_call_and_return_conditional_losses_379716

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_488_layer_call_and_return_conditional_losses_378252

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_3_380588s
Ygcn_attention_30_core_gcn_61_conv2d_123_kernel_regularizer_square_readvariableop_resource:
identity??Pgcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/Square/ReadVariableOp?
Pgcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/Square/ReadVariableOpReadVariableOpYgcn_attention_30_core_gcn_61_conv2d_123_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype02R
Pgcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/Square/ReadVariableOp?
Agcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/SquareSquareXgcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2C
Agcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/Square?
@gcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2B
@gcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/Const?
>gcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/SumSumEgcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/Square:y:0Igcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2@
>gcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/Sum?
@gcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82B
@gcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/mul/x?
>gcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/mulMulIgcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/mul/x:output:0Ggcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2@
>gcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/mul?
IdentityIdentityBgcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOpQ^gcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2?
Pgcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/Square/ReadVariableOpPgcn_attention_30/core_gcn_61/conv2d_123/kernel/Regularizer/Square/ReadVariableOp
?	
?
8__inference_batch_normalization_490_layer_call_fn_380146

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_490_layer_call_and_return_conditional_losses_3784962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?n
__inference_tf_translate_376947
input_text1
input_text2
input_text3^
Ggcn_attention_30_conv1d_240_conv1d_expanddims_1_readvariableop_resource:?I
;gcn_attention_30_conv1d_240_biasadd_readvariableop_resource:S
Egcn_attention_30_batch_normalization_480_cast_readvariableop_resource:U
Ggcn_attention_30_batch_normalization_480_cast_1_readvariableop_resource:U
Ggcn_attention_30_batch_normalization_480_cast_2_readvariableop_resource:U
Ggcn_attention_30_batch_normalization_480_cast_3_readvariableop_resource:M
;gcn_attention_30_dense_30_tensordot_readvariableop_resource:	T
:gcn_attention_30_conv2d_121_conv2d_readvariableop_resource:		I
;gcn_attention_30_conv2d_121_biasadd_readvariableop_resource:	N
@gcn_attention_30_batch_normalization_482_readvariableop_resource:	P
Bgcn_attention_30_batch_normalization_482_readvariableop_1_resource:	_
Qgcn_attention_30_batch_normalization_482_fusedbatchnormv3_readvariableop_resource:	a
Sgcn_attention_30_batch_normalization_482_fusedbatchnormv3_readvariableop_1_resource:	G
5gcn_attention_30_embedding_30_embedding_lookup_376285:	`
Fgcn_attention_30_core_gcn_60_conv2d_122_conv2d_readvariableop_resource:U
Ggcn_attention_30_core_gcn_60_conv2d_122_biasadd_readvariableop_resource:Z
Lgcn_attention_30_core_gcn_60_batch_normalization_484_readvariableop_resource:\
Ngcn_attention_30_core_gcn_60_batch_normalization_484_readvariableop_1_resource:k
]gcn_attention_30_core_gcn_60_batch_normalization_484_fusedbatchnormv3_readvariableop_resource:m
_gcn_attention_30_core_gcn_60_batch_normalization_484_fusedbatchnormv3_readvariableop_1_resource:i
Sgcn_attention_30_core_gcn_60_conv1d_241_conv1d_expanddims_1_readvariableop_resource:U
Ggcn_attention_30_core_gcn_60_conv1d_241_biasadd_readvariableop_resource:_
Qgcn_attention_30_core_gcn_60_batch_normalization_485_cast_readvariableop_resource:a
Sgcn_attention_30_core_gcn_60_batch_normalization_485_cast_1_readvariableop_resource:a
Sgcn_attention_30_core_gcn_60_batch_normalization_485_cast_2_readvariableop_resource:a
Sgcn_attention_30_core_gcn_60_batch_normalization_485_cast_3_readvariableop_resource:i
Sgcn_attention_30_core_gcn_60_conv1d_242_conv1d_expanddims_1_readvariableop_resource:U
Ggcn_attention_30_core_gcn_60_conv1d_242_biasadd_readvariableop_resource:_
Qgcn_attention_30_core_gcn_60_batch_normalization_486_cast_readvariableop_resource:a
Sgcn_attention_30_core_gcn_60_batch_normalization_486_cast_1_readvariableop_resource:a
Sgcn_attention_30_core_gcn_60_batch_normalization_486_cast_2_readvariableop_resource:a
Sgcn_attention_30_core_gcn_60_batch_normalization_486_cast_3_readvariableop_resource:i
Sgcn_attention_30_core_gcn_60_conv1d_243_conv1d_expanddims_1_readvariableop_resource:U
Ggcn_attention_30_core_gcn_60_conv1d_243_biasadd_readvariableop_resource:_
Qgcn_attention_30_core_gcn_60_batch_normalization_487_cast_readvariableop_resource:a
Sgcn_attention_30_core_gcn_60_batch_normalization_487_cast_1_readvariableop_resource:a
Sgcn_attention_30_core_gcn_60_batch_normalization_487_cast_2_readvariableop_resource:a
Sgcn_attention_30_core_gcn_60_batch_normalization_487_cast_3_readvariableop_resource:Z
Lgcn_attention_30_core_gcn_60_batch_normalization_488_readvariableop_resource:\
Ngcn_attention_30_core_gcn_60_batch_normalization_488_readvariableop_1_resource:k
]gcn_attention_30_core_gcn_60_batch_normalization_488_fusedbatchnormv3_readvariableop_resource:m
_gcn_attention_30_core_gcn_60_batch_normalization_488_fusedbatchnormv3_readvariableop_1_resource:_
Qgcn_attention_30_core_gcn_60_batch_normalization_489_cast_readvariableop_resource:a
Sgcn_attention_30_core_gcn_60_batch_normalization_489_cast_1_readvariableop_resource:a
Sgcn_attention_30_core_gcn_60_batch_normalization_489_cast_2_readvariableop_resource:a
Sgcn_attention_30_core_gcn_60_batch_normalization_489_cast_3_readvariableop_resource:`
Fgcn_attention_30_core_gcn_61_conv2d_123_conv2d_readvariableop_resource:U
Ggcn_attention_30_core_gcn_61_conv2d_123_biasadd_readvariableop_resource:Z
Lgcn_attention_30_core_gcn_61_batch_normalization_490_readvariableop_resource:\
Ngcn_attention_30_core_gcn_61_batch_normalization_490_readvariableop_1_resource:k
]gcn_attention_30_core_gcn_61_batch_normalization_490_fusedbatchnormv3_readvariableop_resource:m
_gcn_attention_30_core_gcn_61_batch_normalization_490_fusedbatchnormv3_readvariableop_1_resource:i
Sgcn_attention_30_core_gcn_61_conv1d_244_conv1d_expanddims_1_readvariableop_resource:U
Ggcn_attention_30_core_gcn_61_conv1d_244_biasadd_readvariableop_resource:_
Qgcn_attention_30_core_gcn_61_batch_normalization_491_cast_readvariableop_resource:a
Sgcn_attention_30_core_gcn_61_batch_normalization_491_cast_1_readvariableop_resource:a
Sgcn_attention_30_core_gcn_61_batch_normalization_491_cast_2_readvariableop_resource:a
Sgcn_attention_30_core_gcn_61_batch_normalization_491_cast_3_readvariableop_resource:i
Sgcn_attention_30_core_gcn_61_conv1d_245_conv1d_expanddims_1_readvariableop_resource:U
Ggcn_attention_30_core_gcn_61_conv1d_245_biasadd_readvariableop_resource:_
Qgcn_attention_30_core_gcn_61_batch_normalization_492_cast_readvariableop_resource:a
Sgcn_attention_30_core_gcn_61_batch_normalization_492_cast_1_readvariableop_resource:a
Sgcn_attention_30_core_gcn_61_batch_normalization_492_cast_2_readvariableop_resource:a
Sgcn_attention_30_core_gcn_61_batch_normalization_492_cast_3_readvariableop_resource:i
Sgcn_attention_30_core_gcn_61_conv1d_246_conv1d_expanddims_1_readvariableop_resource:U
Ggcn_attention_30_core_gcn_61_conv1d_246_biasadd_readvariableop_resource:_
Qgcn_attention_30_core_gcn_61_batch_normalization_493_cast_readvariableop_resource:a
Sgcn_attention_30_core_gcn_61_batch_normalization_493_cast_1_readvariableop_resource:a
Sgcn_attention_30_core_gcn_61_batch_normalization_493_cast_2_readvariableop_resource:a
Sgcn_attention_30_core_gcn_61_batch_normalization_493_cast_3_readvariableop_resource:Z
Lgcn_attention_30_core_gcn_61_batch_normalization_494_readvariableop_resource:\
Ngcn_attention_30_core_gcn_61_batch_normalization_494_readvariableop_1_resource:k
]gcn_attention_30_core_gcn_61_batch_normalization_494_fusedbatchnormv3_readvariableop_resource:m
_gcn_attention_30_core_gcn_61_batch_normalization_494_fusedbatchnormv3_readvariableop_1_resource:_
Qgcn_attention_30_core_gcn_61_batch_normalization_495_cast_readvariableop_resource:a
Sgcn_attention_30_core_gcn_61_batch_normalization_495_cast_1_readvariableop_resource:a
Sgcn_attention_30_core_gcn_61_batch_normalization_495_cast_2_readvariableop_resource:a
Sgcn_attention_30_core_gcn_61_batch_normalization_495_cast_3_readvariableop_resource:S
Egcn_attention_30_batch_normalization_483_cast_readvariableop_resource:U
Ggcn_attention_30_batch_normalization_483_cast_1_readvariableop_resource:U
Ggcn_attention_30_batch_normalization_483_cast_2_readvariableop_resource:U
Ggcn_attention_30_batch_normalization_483_cast_3_readvariableop_resource:]
Ggcn_attention_30_conv1d_247_conv1d_expanddims_1_readvariableop_resource:I
;gcn_attention_30_conv1d_247_biasadd_readvariableop_resource:
identity??<gcn_attention_30/batch_normalization_480/Cast/ReadVariableOp?>gcn_attention_30/batch_normalization_480/Cast_1/ReadVariableOp?>gcn_attention_30/batch_normalization_480/Cast_2/ReadVariableOp?>gcn_attention_30/batch_normalization_480/Cast_3/ReadVariableOp?Hgcn_attention_30/batch_normalization_482/FusedBatchNormV3/ReadVariableOp?Jgcn_attention_30/batch_normalization_482/FusedBatchNormV3/ReadVariableOp_1?7gcn_attention_30/batch_normalization_482/ReadVariableOp?9gcn_attention_30/batch_normalization_482/ReadVariableOp_1?<gcn_attention_30/batch_normalization_483/Cast/ReadVariableOp?>gcn_attention_30/batch_normalization_483/Cast_1/ReadVariableOp?>gcn_attention_30/batch_normalization_483/Cast_2/ReadVariableOp?>gcn_attention_30/batch_normalization_483/Cast_3/ReadVariableOp?2gcn_attention_30/conv1d_240/BiasAdd/ReadVariableOp?>gcn_attention_30/conv1d_240/conv1d/ExpandDims_1/ReadVariableOp?2gcn_attention_30/conv1d_247/BiasAdd/ReadVariableOp?>gcn_attention_30/conv1d_247/conv1d/ExpandDims_1/ReadVariableOp?2gcn_attention_30/conv2d_121/BiasAdd/ReadVariableOp?1gcn_attention_30/conv2d_121/Conv2D/ReadVariableOp?Tgcn_attention_30/core_gcn_60/batch_normalization_484/FusedBatchNormV3/ReadVariableOp?Vgcn_attention_30/core_gcn_60/batch_normalization_484/FusedBatchNormV3/ReadVariableOp_1?Cgcn_attention_30/core_gcn_60/batch_normalization_484/ReadVariableOp?Egcn_attention_30/core_gcn_60/batch_normalization_484/ReadVariableOp_1?Hgcn_attention_30/core_gcn_60/batch_normalization_485/Cast/ReadVariableOp?Jgcn_attention_30/core_gcn_60/batch_normalization_485/Cast_1/ReadVariableOp?Jgcn_attention_30/core_gcn_60/batch_normalization_485/Cast_2/ReadVariableOp?Jgcn_attention_30/core_gcn_60/batch_normalization_485/Cast_3/ReadVariableOp?Hgcn_attention_30/core_gcn_60/batch_normalization_486/Cast/ReadVariableOp?Jgcn_attention_30/core_gcn_60/batch_normalization_486/Cast_1/ReadVariableOp?Jgcn_attention_30/core_gcn_60/batch_normalization_486/Cast_2/ReadVariableOp?Jgcn_attention_30/core_gcn_60/batch_normalization_486/Cast_3/ReadVariableOp?Hgcn_attention_30/core_gcn_60/batch_normalization_487/Cast/ReadVariableOp?Jgcn_attention_30/core_gcn_60/batch_normalization_487/Cast_1/ReadVariableOp?Jgcn_attention_30/core_gcn_60/batch_normalization_487/Cast_2/ReadVariableOp?Jgcn_attention_30/core_gcn_60/batch_normalization_487/Cast_3/ReadVariableOp?Tgcn_attention_30/core_gcn_60/batch_normalization_488/FusedBatchNormV3/ReadVariableOp?Vgcn_attention_30/core_gcn_60/batch_normalization_488/FusedBatchNormV3/ReadVariableOp_1?Cgcn_attention_30/core_gcn_60/batch_normalization_488/ReadVariableOp?Egcn_attention_30/core_gcn_60/batch_normalization_488/ReadVariableOp_1?Hgcn_attention_30/core_gcn_60/batch_normalization_489/Cast/ReadVariableOp?Jgcn_attention_30/core_gcn_60/batch_normalization_489/Cast_1/ReadVariableOp?Jgcn_attention_30/core_gcn_60/batch_normalization_489/Cast_2/ReadVariableOp?Jgcn_attention_30/core_gcn_60/batch_normalization_489/Cast_3/ReadVariableOp?>gcn_attention_30/core_gcn_60/conv1d_241/BiasAdd/ReadVariableOp?Jgcn_attention_30/core_gcn_60/conv1d_241/conv1d/ExpandDims_1/ReadVariableOp?>gcn_attention_30/core_gcn_60/conv1d_242/BiasAdd/ReadVariableOp?Jgcn_attention_30/core_gcn_60/conv1d_242/conv1d/ExpandDims_1/ReadVariableOp?>gcn_attention_30/core_gcn_60/conv1d_243/BiasAdd/ReadVariableOp?Jgcn_attention_30/core_gcn_60/conv1d_243/conv1d/ExpandDims_1/ReadVariableOp?>gcn_attention_30/core_gcn_60/conv2d_122/BiasAdd/ReadVariableOp?=gcn_attention_30/core_gcn_60/conv2d_122/Conv2D/ReadVariableOp?Tgcn_attention_30/core_gcn_61/batch_normalization_490/FusedBatchNormV3/ReadVariableOp?Vgcn_attention_30/core_gcn_61/batch_normalization_490/FusedBatchNormV3/ReadVariableOp_1?Cgcn_attention_30/core_gcn_61/batch_normalization_490/ReadVariableOp?Egcn_attention_30/core_gcn_61/batch_normalization_490/ReadVariableOp_1?Hgcn_attention_30/core_gcn_61/batch_normalization_491/Cast/ReadVariableOp?Jgcn_attention_30/core_gcn_61/batch_normalization_491/Cast_1/ReadVariableOp?Jgcn_attention_30/core_gcn_61/batch_normalization_491/Cast_2/ReadVariableOp?Jgcn_attention_30/core_gcn_61/batch_normalization_491/Cast_3/ReadVariableOp?Hgcn_attention_30/core_gcn_61/batch_normalization_492/Cast/ReadVariableOp?Jgcn_attention_30/core_gcn_61/batch_normalization_492/Cast_1/ReadVariableOp?Jgcn_attention_30/core_gcn_61/batch_normalization_492/Cast_2/ReadVariableOp?Jgcn_attention_30/core_gcn_61/batch_normalization_492/Cast_3/ReadVariableOp?Hgcn_attention_30/core_gcn_61/batch_normalization_493/Cast/ReadVariableOp?Jgcn_attention_30/core_gcn_61/batch_normalization_493/Cast_1/ReadVariableOp?Jgcn_attention_30/core_gcn_61/batch_normalization_493/Cast_2/ReadVariableOp?Jgcn_attention_30/core_gcn_61/batch_normalization_493/Cast_3/ReadVariableOp?Tgcn_attention_30/core_gcn_61/batch_normalization_494/FusedBatchNormV3/ReadVariableOp?Vgcn_attention_30/core_gcn_61/batch_normalization_494/FusedBatchNormV3/ReadVariableOp_1?Cgcn_attention_30/core_gcn_61/batch_normalization_494/ReadVariableOp?Egcn_attention_30/core_gcn_61/batch_normalization_494/ReadVariableOp_1?Hgcn_attention_30/core_gcn_61/batch_normalization_495/Cast/ReadVariableOp?Jgcn_attention_30/core_gcn_61/batch_normalization_495/Cast_1/ReadVariableOp?Jgcn_attention_30/core_gcn_61/batch_normalization_495/Cast_2/ReadVariableOp?Jgcn_attention_30/core_gcn_61/batch_normalization_495/Cast_3/ReadVariableOp?>gcn_attention_30/core_gcn_61/conv1d_244/BiasAdd/ReadVariableOp?Jgcn_attention_30/core_gcn_61/conv1d_244/conv1d/ExpandDims_1/ReadVariableOp?>gcn_attention_30/core_gcn_61/conv1d_245/BiasAdd/ReadVariableOp?Jgcn_attention_30/core_gcn_61/conv1d_245/conv1d/ExpandDims_1/ReadVariableOp?>gcn_attention_30/core_gcn_61/conv1d_246/BiasAdd/ReadVariableOp?Jgcn_attention_30/core_gcn_61/conv1d_246/conv1d/ExpandDims_1/ReadVariableOp?>gcn_attention_30/core_gcn_61/conv2d_123/BiasAdd/ReadVariableOp?=gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/ReadVariableOp?2gcn_attention_30/dense_30/Tensordot/ReadVariableOp?.gcn_attention_30/embedding_30/embedding_lookup?
1gcn_attention_30/conv1d_240/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????23
1gcn_attention_30/conv1d_240/conv1d/ExpandDims/dim?
-gcn_attention_30/conv1d_240/conv1d/ExpandDims
ExpandDimsinput_text1:gcn_attention_30/conv1d_240/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2/
-gcn_attention_30/conv1d_240/conv1d/ExpandDims?
>gcn_attention_30/conv1d_240/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpGgcn_attention_30_conv1d_240_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?*
dtype02@
>gcn_attention_30/conv1d_240/conv1d/ExpandDims_1/ReadVariableOp?
3gcn_attention_30/conv1d_240/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3gcn_attention_30/conv1d_240/conv1d/ExpandDims_1/dim?
/gcn_attention_30/conv1d_240/conv1d/ExpandDims_1
ExpandDimsFgcn_attention_30/conv1d_240/conv1d/ExpandDims_1/ReadVariableOp:value:0<gcn_attention_30/conv1d_240/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?21
/gcn_attention_30/conv1d_240/conv1d/ExpandDims_1?
"gcn_attention_30/conv1d_240/conv1dConv2D6gcn_attention_30/conv1d_240/conv1d/ExpandDims:output:08gcn_attention_30/conv1d_240/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
2$
"gcn_attention_30/conv1d_240/conv1d?
*gcn_attention_30/conv1d_240/conv1d/SqueezeSqueeze+gcn_attention_30/conv1d_240/conv1d:output:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims

?????????2,
*gcn_attention_30/conv1d_240/conv1d/Squeeze?
2gcn_attention_30/conv1d_240/BiasAdd/ReadVariableOpReadVariableOp;gcn_attention_30_conv1d_240_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2gcn_attention_30/conv1d_240/BiasAdd/ReadVariableOp?
#gcn_attention_30/conv1d_240/BiasAddBiasAdd3gcn_attention_30/conv1d_240/conv1d/Squeeze:output:0:gcn_attention_30/conv1d_240/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2%
#gcn_attention_30/conv1d_240/BiasAdd?
$gcn_attention_30/activation_150/ReluRelu,gcn_attention_30/conv1d_240/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????2&
$gcn_attention_30/activation_150/Relu?
<gcn_attention_30/batch_normalization_480/Cast/ReadVariableOpReadVariableOpEgcn_attention_30_batch_normalization_480_cast_readvariableop_resource*
_output_shapes
:*
dtype02>
<gcn_attention_30/batch_normalization_480/Cast/ReadVariableOp?
>gcn_attention_30/batch_normalization_480/Cast_1/ReadVariableOpReadVariableOpGgcn_attention_30_batch_normalization_480_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_30/batch_normalization_480/Cast_1/ReadVariableOp?
>gcn_attention_30/batch_normalization_480/Cast_2/ReadVariableOpReadVariableOpGgcn_attention_30_batch_normalization_480_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_30/batch_normalization_480/Cast_2/ReadVariableOp?
>gcn_attention_30/batch_normalization_480/Cast_3/ReadVariableOpReadVariableOpGgcn_attention_30_batch_normalization_480_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_30/batch_normalization_480/Cast_3/ReadVariableOp?
8gcn_attention_30/batch_normalization_480/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2:
8gcn_attention_30/batch_normalization_480/batchnorm/add/y?
6gcn_attention_30/batch_normalization_480/batchnorm/addAddV2Fgcn_attention_30/batch_normalization_480/Cast_1/ReadVariableOp:value:0Agcn_attention_30/batch_normalization_480/batchnorm/add/y:output:0*
T0*
_output_shapes
:28
6gcn_attention_30/batch_normalization_480/batchnorm/add?
8gcn_attention_30/batch_normalization_480/batchnorm/RsqrtRsqrt:gcn_attention_30/batch_normalization_480/batchnorm/add:z:0*
T0*
_output_shapes
:2:
8gcn_attention_30/batch_normalization_480/batchnorm/Rsqrt?
6gcn_attention_30/batch_normalization_480/batchnorm/mulMul<gcn_attention_30/batch_normalization_480/batchnorm/Rsqrt:y:0Fgcn_attention_30/batch_normalization_480/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:28
6gcn_attention_30/batch_normalization_480/batchnorm/mul?
8gcn_attention_30/batch_normalization_480/batchnorm/mul_1Mul2gcn_attention_30/activation_150/Relu:activations:0:gcn_attention_30/batch_normalization_480/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2:
8gcn_attention_30/batch_normalization_480/batchnorm/mul_1?
8gcn_attention_30/batch_normalization_480/batchnorm/mul_2MulDgcn_attention_30/batch_normalization_480/Cast/ReadVariableOp:value:0:gcn_attention_30/batch_normalization_480/batchnorm/mul:z:0*
T0*
_output_shapes
:2:
8gcn_attention_30/batch_normalization_480/batchnorm/mul_2?
6gcn_attention_30/batch_normalization_480/batchnorm/subSubFgcn_attention_30/batch_normalization_480/Cast_2/ReadVariableOp:value:0<gcn_attention_30/batch_normalization_480/batchnorm/mul_2:z:0*
T0*
_output_shapes
:28
6gcn_attention_30/batch_normalization_480/batchnorm/sub?
8gcn_attention_30/batch_normalization_480/batchnorm/add_1AddV2<gcn_attention_30/batch_normalization_480/batchnorm/mul_1:z:0:gcn_attention_30/batch_normalization_480/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2:
8gcn_attention_30/batch_normalization_480/batchnorm/add_1?
%gcn_attention_30/dropout_360/IdentityIdentity<gcn_attention_30/batch_normalization_480/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????2'
%gcn_attention_30/dropout_360/Identity?
2gcn_attention_30/dense_30/Tensordot/ReadVariableOpReadVariableOp;gcn_attention_30_dense_30_tensordot_readvariableop_resource*
_output_shapes

:	*
dtype024
2gcn_attention_30/dense_30/Tensordot/ReadVariableOp?
(gcn_attention_30/dense_30/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2*
(gcn_attention_30/dense_30/Tensordot/axes?
(gcn_attention_30/dense_30/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(gcn_attention_30/dense_30/Tensordot/free?
)gcn_attention_30/dense_30/Tensordot/ShapeShapeinput_text3*
T0*
_output_shapes
:2+
)gcn_attention_30/dense_30/Tensordot/Shape?
1gcn_attention_30/dense_30/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1gcn_attention_30/dense_30/Tensordot/GatherV2/axis?
,gcn_attention_30/dense_30/Tensordot/GatherV2GatherV22gcn_attention_30/dense_30/Tensordot/Shape:output:01gcn_attention_30/dense_30/Tensordot/free:output:0:gcn_attention_30/dense_30/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,gcn_attention_30/dense_30/Tensordot/GatherV2?
3gcn_attention_30/dense_30/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 25
3gcn_attention_30/dense_30/Tensordot/GatherV2_1/axis?
.gcn_attention_30/dense_30/Tensordot/GatherV2_1GatherV22gcn_attention_30/dense_30/Tensordot/Shape:output:01gcn_attention_30/dense_30/Tensordot/axes:output:0<gcn_attention_30/dense_30/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:20
.gcn_attention_30/dense_30/Tensordot/GatherV2_1?
)gcn_attention_30/dense_30/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)gcn_attention_30/dense_30/Tensordot/Const?
(gcn_attention_30/dense_30/Tensordot/ProdProd5gcn_attention_30/dense_30/Tensordot/GatherV2:output:02gcn_attention_30/dense_30/Tensordot/Const:output:0*
T0*
_output_shapes
: 2*
(gcn_attention_30/dense_30/Tensordot/Prod?
+gcn_attention_30/dense_30/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+gcn_attention_30/dense_30/Tensordot/Const_1?
*gcn_attention_30/dense_30/Tensordot/Prod_1Prod7gcn_attention_30/dense_30/Tensordot/GatherV2_1:output:04gcn_attention_30/dense_30/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2,
*gcn_attention_30/dense_30/Tensordot/Prod_1?
/gcn_attention_30/dense_30/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/gcn_attention_30/dense_30/Tensordot/concat/axis?
*gcn_attention_30/dense_30/Tensordot/concatConcatV21gcn_attention_30/dense_30/Tensordot/free:output:01gcn_attention_30/dense_30/Tensordot/axes:output:08gcn_attention_30/dense_30/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*gcn_attention_30/dense_30/Tensordot/concat?
)gcn_attention_30/dense_30/Tensordot/stackPack1gcn_attention_30/dense_30/Tensordot/Prod:output:03gcn_attention_30/dense_30/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2+
)gcn_attention_30/dense_30/Tensordot/stack?
-gcn_attention_30/dense_30/Tensordot/transpose	Transposeinput_text33gcn_attention_30/dense_30/Tensordot/concat:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2/
-gcn_attention_30/dense_30/Tensordot/transpose?
+gcn_attention_30/dense_30/Tensordot/ReshapeReshape1gcn_attention_30/dense_30/Tensordot/transpose:y:02gcn_attention_30/dense_30/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2-
+gcn_attention_30/dense_30/Tensordot/Reshape?
*gcn_attention_30/dense_30/Tensordot/MatMulMatMul4gcn_attention_30/dense_30/Tensordot/Reshape:output:0:gcn_attention_30/dense_30/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2,
*gcn_attention_30/dense_30/Tensordot/MatMul?
+gcn_attention_30/dense_30/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	2-
+gcn_attention_30/dense_30/Tensordot/Const_2?
1gcn_attention_30/dense_30/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1gcn_attention_30/dense_30/Tensordot/concat_1/axis?
,gcn_attention_30/dense_30/Tensordot/concat_1ConcatV25gcn_attention_30/dense_30/Tensordot/GatherV2:output:04gcn_attention_30/dense_30/Tensordot/Const_2:output:0:gcn_attention_30/dense_30/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2.
,gcn_attention_30/dense_30/Tensordot/concat_1?
#gcn_attention_30/dense_30/TensordotReshape4gcn_attention_30/dense_30/Tensordot/MatMul:product:05gcn_attention_30/dense_30/Tensordot/concat_1:output:0*
T0*A
_output_shapes/
-:+???????????????????????????	2%
#gcn_attention_30/dense_30/Tensordot?
1gcn_attention_30/conv2d_121/Conv2D/ReadVariableOpReadVariableOp:gcn_attention_30_conv2d_121_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype023
1gcn_attention_30/conv2d_121/Conv2D/ReadVariableOp?
"gcn_attention_30/conv2d_121/Conv2DConv2D,gcn_attention_30/dense_30/Tensordot:output:09gcn_attention_30/conv2d_121/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????	*
paddingSAME*
strides
2$
"gcn_attention_30/conv2d_121/Conv2D?
2gcn_attention_30/conv2d_121/BiasAdd/ReadVariableOpReadVariableOp;gcn_attention_30_conv2d_121_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype024
2gcn_attention_30/conv2d_121/BiasAdd/ReadVariableOp?
#gcn_attention_30/conv2d_121/BiasAddBiasAdd+gcn_attention_30/conv2d_121/Conv2D:output:0:gcn_attention_30/conv2d_121/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????	2%
#gcn_attention_30/conv2d_121/BiasAdd?
$gcn_attention_30/activation_152/ReluRelu,gcn_attention_30/conv2d_121/BiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????	2&
$gcn_attention_30/activation_152/Relu?
7gcn_attention_30/batch_normalization_482/ReadVariableOpReadVariableOp@gcn_attention_30_batch_normalization_482_readvariableop_resource*
_output_shapes
:	*
dtype029
7gcn_attention_30/batch_normalization_482/ReadVariableOp?
9gcn_attention_30/batch_normalization_482/ReadVariableOp_1ReadVariableOpBgcn_attention_30_batch_normalization_482_readvariableop_1_resource*
_output_shapes
:	*
dtype02;
9gcn_attention_30/batch_normalization_482/ReadVariableOp_1?
Hgcn_attention_30/batch_normalization_482/FusedBatchNormV3/ReadVariableOpReadVariableOpQgcn_attention_30_batch_normalization_482_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:	*
dtype02J
Hgcn_attention_30/batch_normalization_482/FusedBatchNormV3/ReadVariableOp?
Jgcn_attention_30/batch_normalization_482/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpSgcn_attention_30_batch_normalization_482_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:	*
dtype02L
Jgcn_attention_30/batch_normalization_482/FusedBatchNormV3/ReadVariableOp_1?
9gcn_attention_30/batch_normalization_482/FusedBatchNormV3FusedBatchNormV32gcn_attention_30/activation_152/Relu:activations:0?gcn_attention_30/batch_normalization_482/ReadVariableOp:value:0Agcn_attention_30/batch_normalization_482/ReadVariableOp_1:value:0Pgcn_attention_30/batch_normalization_482/FusedBatchNormV3/ReadVariableOp:value:0Rgcn_attention_30/batch_normalization_482/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????	:	:	:	:	:*
epsilon%o?:*
is_training( 2;
9gcn_attention_30/batch_normalization_482/FusedBatchNormV3?
%gcn_attention_30/dropout_362/IdentityIdentity=gcn_attention_30/batch_normalization_482/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????	2'
%gcn_attention_30/dropout_362/Identity?
gcn_attention_30/SqueezeSqueezeinput_text2*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims

?????????2
gcn_attention_30/Squeeze?
"gcn_attention_30/embedding_30/CastCast!gcn_attention_30/Squeeze:output:0*

DstT0*

SrcT0*=
_output_shapes+
):'???????????????????????????2$
"gcn_attention_30/embedding_30/Cast?
.gcn_attention_30/embedding_30/embedding_lookupResourceGather5gcn_attention_30_embedding_30_embedding_lookup_376285&gcn_attention_30/embedding_30/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*H
_class>
<:loc:@gcn_attention_30/embedding_30/embedding_lookup/376285*A
_output_shapes/
-:+???????????????????????????	*
dtype020
.gcn_attention_30/embedding_30/embedding_lookup?
7gcn_attention_30/embedding_30/embedding_lookup/IdentityIdentity7gcn_attention_30/embedding_30/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*H
_class>
<:loc:@gcn_attention_30/embedding_30/embedding_lookup/376285*A
_output_shapes/
-:+???????????????????????????	29
7gcn_attention_30/embedding_30/embedding_lookup/Identity?
9gcn_attention_30/embedding_30/embedding_lookup/Identity_1Identity@gcn_attention_30/embedding_30/embedding_lookup/Identity:output:0*
T0*A
_output_shapes/
-:+???????????????????????????	2;
9gcn_attention_30/embedding_30/embedding_lookup/Identity_1?
(gcn_attention_30/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2*
(gcn_attention_30/concatenate/concat/axis?
#gcn_attention_30/concatenate/concatConcatV2.gcn_attention_30/dropout_362/Identity:output:0Bgcn_attention_30/embedding_30/embedding_lookup/Identity_1:output:01gcn_attention_30/concatenate/concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+???????????????????????????2%
#gcn_attention_30/concatenate/concat?
=gcn_attention_30/core_gcn_60/conv2d_122/Conv2D/ReadVariableOpReadVariableOpFgcn_attention_30_core_gcn_60_conv2d_122_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02?
=gcn_attention_30/core_gcn_60/conv2d_122/Conv2D/ReadVariableOp?
.gcn_attention_30/core_gcn_60/conv2d_122/Conv2DConv2D,gcn_attention_30/concatenate/concat:output:0Egcn_attention_30/core_gcn_60/conv2d_122/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
20
.gcn_attention_30/core_gcn_60/conv2d_122/Conv2D?
>gcn_attention_30/core_gcn_60/conv2d_122/BiasAdd/ReadVariableOpReadVariableOpGgcn_attention_30_core_gcn_60_conv2d_122_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_30/core_gcn_60/conv2d_122/BiasAdd/ReadVariableOp?
/gcn_attention_30/core_gcn_60/conv2d_122/BiasAddBiasAdd7gcn_attention_30/core_gcn_60/conv2d_122/Conv2D:output:0Fgcn_attention_30/core_gcn_60/conv2d_122/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????21
/gcn_attention_30/core_gcn_60/conv2d_122/BiasAdd?
,gcn_attention_30/core_gcn_60/activation/ReluRelu8gcn_attention_30/core_gcn_60/conv2d_122/BiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2.
,gcn_attention_30/core_gcn_60/activation/Relu?
Cgcn_attention_30/core_gcn_60/batch_normalization_484/ReadVariableOpReadVariableOpLgcn_attention_30_core_gcn_60_batch_normalization_484_readvariableop_resource*
_output_shapes
:*
dtype02E
Cgcn_attention_30/core_gcn_60/batch_normalization_484/ReadVariableOp?
Egcn_attention_30/core_gcn_60/batch_normalization_484/ReadVariableOp_1ReadVariableOpNgcn_attention_30_core_gcn_60_batch_normalization_484_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Egcn_attention_30/core_gcn_60/batch_normalization_484/ReadVariableOp_1?
Tgcn_attention_30/core_gcn_60/batch_normalization_484/FusedBatchNormV3/ReadVariableOpReadVariableOp]gcn_attention_30_core_gcn_60_batch_normalization_484_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02V
Tgcn_attention_30/core_gcn_60/batch_normalization_484/FusedBatchNormV3/ReadVariableOp?
Vgcn_attention_30/core_gcn_60/batch_normalization_484/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_gcn_attention_30_core_gcn_60_batch_normalization_484_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02X
Vgcn_attention_30/core_gcn_60/batch_normalization_484/FusedBatchNormV3/ReadVariableOp_1?
Egcn_attention_30/core_gcn_60/batch_normalization_484/FusedBatchNormV3FusedBatchNormV3:gcn_attention_30/core_gcn_60/activation/Relu:activations:0Kgcn_attention_30/core_gcn_60/batch_normalization_484/ReadVariableOp:value:0Mgcn_attention_30/core_gcn_60/batch_normalization_484/ReadVariableOp_1:value:0\gcn_attention_30/core_gcn_60/batch_normalization_484/FusedBatchNormV3/ReadVariableOp:value:0^gcn_attention_30/core_gcn_60/batch_normalization_484/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2G
Egcn_attention_30/core_gcn_60/batch_normalization_484/FusedBatchNormV3?
1gcn_attention_30/core_gcn_60/dropout_364/IdentityIdentityIgcn_attention_30/core_gcn_60/batch_normalization_484/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????23
1gcn_attention_30/core_gcn_60/dropout_364/Identity?
=gcn_attention_30/core_gcn_60/conv1d_241/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2?
=gcn_attention_30/core_gcn_60/conv1d_241/conv1d/ExpandDims/dim?
9gcn_attention_30/core_gcn_60/conv1d_241/conv1d/ExpandDims
ExpandDims.gcn_attention_30/dropout_360/Identity:output:0Fgcn_attention_30/core_gcn_60/conv1d_241/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2;
9gcn_attention_30/core_gcn_60/conv1d_241/conv1d/ExpandDims?
Jgcn_attention_30/core_gcn_60/conv1d_241/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_60_conv1d_241_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_60/conv1d_241/conv1d/ExpandDims_1/ReadVariableOp?
?gcn_attention_30/core_gcn_60/conv1d_241/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2A
?gcn_attention_30/core_gcn_60/conv1d_241/conv1d/ExpandDims_1/dim?
;gcn_attention_30/core_gcn_60/conv1d_241/conv1d/ExpandDims_1
ExpandDimsRgcn_attention_30/core_gcn_60/conv1d_241/conv1d/ExpandDims_1/ReadVariableOp:value:0Hgcn_attention_30/core_gcn_60/conv1d_241/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2=
;gcn_attention_30/core_gcn_60/conv1d_241/conv1d/ExpandDims_1?
.gcn_attention_30/core_gcn_60/conv1d_241/conv1dConv2DBgcn_attention_30/core_gcn_60/conv1d_241/conv1d/ExpandDims:output:0Dgcn_attention_30/core_gcn_60/conv1d_241/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
20
.gcn_attention_30/core_gcn_60/conv1d_241/conv1d?
6gcn_attention_30/core_gcn_60/conv1d_241/conv1d/SqueezeSqueeze7gcn_attention_30/core_gcn_60/conv1d_241/conv1d:output:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims

?????????28
6gcn_attention_30/core_gcn_60/conv1d_241/conv1d/Squeeze?
>gcn_attention_30/core_gcn_60/conv1d_241/BiasAdd/ReadVariableOpReadVariableOpGgcn_attention_30_core_gcn_60_conv1d_241_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_30/core_gcn_60/conv1d_241/BiasAdd/ReadVariableOp?
/gcn_attention_30/core_gcn_60/conv1d_241/BiasAddBiasAdd?gcn_attention_30/core_gcn_60/conv1d_241/conv1d/Squeeze:output:0Fgcn_attention_30/core_gcn_60/conv1d_241/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????21
/gcn_attention_30/core_gcn_60/conv1d_241/BiasAdd?
.gcn_attention_30/core_gcn_60/activation_1/ReluRelu8gcn_attention_30/core_gcn_60/conv1d_241/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????20
.gcn_attention_30/core_gcn_60/activation_1/Relu?
Hgcn_attention_30/core_gcn_60/batch_normalization_485/Cast/ReadVariableOpReadVariableOpQgcn_attention_30_core_gcn_60_batch_normalization_485_cast_readvariableop_resource*
_output_shapes
:*
dtype02J
Hgcn_attention_30/core_gcn_60/batch_normalization_485/Cast/ReadVariableOp?
Jgcn_attention_30/core_gcn_60/batch_normalization_485/Cast_1/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_60_batch_normalization_485_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_60/batch_normalization_485/Cast_1/ReadVariableOp?
Jgcn_attention_30/core_gcn_60/batch_normalization_485/Cast_2/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_60_batch_normalization_485_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_60/batch_normalization_485/Cast_2/ReadVariableOp?
Jgcn_attention_30/core_gcn_60/batch_normalization_485/Cast_3/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_60_batch_normalization_485_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_60/batch_normalization_485/Cast_3/ReadVariableOp?
Dgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2F
Dgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/add/y?
Bgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/addAddV2Rgcn_attention_30/core_gcn_60/batch_normalization_485/Cast_1/ReadVariableOp:value:0Mgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/add/y:output:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/add?
Dgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/RsqrtRsqrtFgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/add:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/Rsqrt?
Bgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/mulMulHgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/Rsqrt:y:0Rgcn_attention_30/core_gcn_60/batch_normalization_485/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/mul?
Dgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/mul_1Mul<gcn_attention_30/core_gcn_60/activation_1/Relu:activations:0Fgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2F
Dgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/mul_1?
Dgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/mul_2MulPgcn_attention_30/core_gcn_60/batch_normalization_485/Cast/ReadVariableOp:value:0Fgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/mul:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/mul_2?
Bgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/subSubRgcn_attention_30/core_gcn_60/batch_normalization_485/Cast_2/ReadVariableOp:value:0Hgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/sub?
Dgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/add_1AddV2Hgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/mul_1:z:0Fgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2F
Dgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/add_1?
1gcn_attention_30/core_gcn_60/dropout_365/IdentityIdentityHgcn_attention_30/core_gcn_60/batch_normalization_485/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????23
1gcn_attention_30/core_gcn_60/dropout_365/Identity?
+gcn_attention_30/core_gcn_60/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+gcn_attention_30/core_gcn_60/ExpandDims/dim?
'gcn_attention_30/core_gcn_60/ExpandDims
ExpandDims:gcn_attention_30/core_gcn_60/dropout_365/Identity:output:04gcn_attention_30/core_gcn_60/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'gcn_attention_30/core_gcn_60/ExpandDims?
-gcn_attention_30/core_gcn_60/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-gcn_attention_30/core_gcn_60/ExpandDims_1/dim?
)gcn_attention_30/core_gcn_60/ExpandDims_1
ExpandDims:gcn_attention_30/core_gcn_60/dropout_365/Identity:output:06gcn_attention_30/core_gcn_60/ExpandDims_1/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2+
)gcn_attention_30/core_gcn_60/ExpandDims_1?
$gcn_attention_30/core_gcn_60/add/addAddV2:gcn_attention_30/core_gcn_60/dropout_364/Identity:output:02gcn_attention_30/core_gcn_60/ExpandDims_1:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2&
$gcn_attention_30/core_gcn_60/add/add?
&gcn_attention_30/core_gcn_60/add/add_1AddV2(gcn_attention_30/core_gcn_60/add/add:z:00gcn_attention_30/core_gcn_60/ExpandDims:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2(
&gcn_attention_30/core_gcn_60/add/add_1?
1gcn_attention_30/core_gcn_60/activation_2/SigmoidSigmoid*gcn_attention_30/core_gcn_60/add/add_1:z:0*
T0*A
_output_shapes/
-:+???????????????????????????23
1gcn_attention_30/core_gcn_60/activation_2/Sigmoid?
=gcn_attention_30/core_gcn_60/conv1d_242/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2?
=gcn_attention_30/core_gcn_60/conv1d_242/conv1d/ExpandDims/dim?
9gcn_attention_30/core_gcn_60/conv1d_242/conv1d/ExpandDims
ExpandDims.gcn_attention_30/dropout_360/Identity:output:0Fgcn_attention_30/core_gcn_60/conv1d_242/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2;
9gcn_attention_30/core_gcn_60/conv1d_242/conv1d/ExpandDims?
Jgcn_attention_30/core_gcn_60/conv1d_242/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_60_conv1d_242_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_60/conv1d_242/conv1d/ExpandDims_1/ReadVariableOp?
?gcn_attention_30/core_gcn_60/conv1d_242/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2A
?gcn_attention_30/core_gcn_60/conv1d_242/conv1d/ExpandDims_1/dim?
;gcn_attention_30/core_gcn_60/conv1d_242/conv1d/ExpandDims_1
ExpandDimsRgcn_attention_30/core_gcn_60/conv1d_242/conv1d/ExpandDims_1/ReadVariableOp:value:0Hgcn_attention_30/core_gcn_60/conv1d_242/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2=
;gcn_attention_30/core_gcn_60/conv1d_242/conv1d/ExpandDims_1?
.gcn_attention_30/core_gcn_60/conv1d_242/conv1dConv2DBgcn_attention_30/core_gcn_60/conv1d_242/conv1d/ExpandDims:output:0Dgcn_attention_30/core_gcn_60/conv1d_242/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
20
.gcn_attention_30/core_gcn_60/conv1d_242/conv1d?
6gcn_attention_30/core_gcn_60/conv1d_242/conv1d/SqueezeSqueeze7gcn_attention_30/core_gcn_60/conv1d_242/conv1d:output:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims

?????????28
6gcn_attention_30/core_gcn_60/conv1d_242/conv1d/Squeeze?
>gcn_attention_30/core_gcn_60/conv1d_242/BiasAdd/ReadVariableOpReadVariableOpGgcn_attention_30_core_gcn_60_conv1d_242_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_30/core_gcn_60/conv1d_242/BiasAdd/ReadVariableOp?
/gcn_attention_30/core_gcn_60/conv1d_242/BiasAddBiasAdd?gcn_attention_30/core_gcn_60/conv1d_242/conv1d/Squeeze:output:0Fgcn_attention_30/core_gcn_60/conv1d_242/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????21
/gcn_attention_30/core_gcn_60/conv1d_242/BiasAdd?
Hgcn_attention_30/core_gcn_60/batch_normalization_486/Cast/ReadVariableOpReadVariableOpQgcn_attention_30_core_gcn_60_batch_normalization_486_cast_readvariableop_resource*
_output_shapes
:*
dtype02J
Hgcn_attention_30/core_gcn_60/batch_normalization_486/Cast/ReadVariableOp?
Jgcn_attention_30/core_gcn_60/batch_normalization_486/Cast_1/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_60_batch_normalization_486_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_60/batch_normalization_486/Cast_1/ReadVariableOp?
Jgcn_attention_30/core_gcn_60/batch_normalization_486/Cast_2/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_60_batch_normalization_486_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_60/batch_normalization_486/Cast_2/ReadVariableOp?
Jgcn_attention_30/core_gcn_60/batch_normalization_486/Cast_3/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_60_batch_normalization_486_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_60/batch_normalization_486/Cast_3/ReadVariableOp?
Dgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2F
Dgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/add/y?
Bgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/addAddV2Rgcn_attention_30/core_gcn_60/batch_normalization_486/Cast_1/ReadVariableOp:value:0Mgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/add/y:output:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/add?
Dgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/RsqrtRsqrtFgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/add:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/Rsqrt?
Bgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/mulMulHgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/Rsqrt:y:0Rgcn_attention_30/core_gcn_60/batch_normalization_486/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/mul?
Dgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/mul_1Mul8gcn_attention_30/core_gcn_60/conv1d_242/BiasAdd:output:0Fgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2F
Dgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/mul_1?
Dgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/mul_2MulPgcn_attention_30/core_gcn_60/batch_normalization_486/Cast/ReadVariableOp:value:0Fgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/mul:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/mul_2?
Bgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/subSubRgcn_attention_30/core_gcn_60/batch_normalization_486/Cast_2/ReadVariableOp:value:0Hgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/sub?
Dgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/add_1AddV2Hgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/mul_1:z:0Fgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2F
Dgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/add_1?
.gcn_attention_30/core_gcn_60/activation_3/ReluReluHgcn_attention_30/core_gcn_60/batch_normalization_486/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????20
.gcn_attention_30/core_gcn_60/activation_3/Relu?
1gcn_attention_30/core_gcn_60/dropout_366/IdentityIdentity<gcn_attention_30/core_gcn_60/activation_3/Relu:activations:0*
T0*4
_output_shapes"
 :??????????????????23
1gcn_attention_30/core_gcn_60/dropout_366/Identity?
=gcn_attention_30/core_gcn_60/conv1d_243/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2?
=gcn_attention_30/core_gcn_60/conv1d_243/conv1d/ExpandDims/dim?
9gcn_attention_30/core_gcn_60/conv1d_243/conv1d/ExpandDims
ExpandDims.gcn_attention_30/dropout_360/Identity:output:0Fgcn_attention_30/core_gcn_60/conv1d_243/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2;
9gcn_attention_30/core_gcn_60/conv1d_243/conv1d/ExpandDims?
Jgcn_attention_30/core_gcn_60/conv1d_243/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_60_conv1d_243_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_60/conv1d_243/conv1d/ExpandDims_1/ReadVariableOp?
?gcn_attention_30/core_gcn_60/conv1d_243/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2A
?gcn_attention_30/core_gcn_60/conv1d_243/conv1d/ExpandDims_1/dim?
;gcn_attention_30/core_gcn_60/conv1d_243/conv1d/ExpandDims_1
ExpandDimsRgcn_attention_30/core_gcn_60/conv1d_243/conv1d/ExpandDims_1/ReadVariableOp:value:0Hgcn_attention_30/core_gcn_60/conv1d_243/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2=
;gcn_attention_30/core_gcn_60/conv1d_243/conv1d/ExpandDims_1?
.gcn_attention_30/core_gcn_60/conv1d_243/conv1dConv2DBgcn_attention_30/core_gcn_60/conv1d_243/conv1d/ExpandDims:output:0Dgcn_attention_30/core_gcn_60/conv1d_243/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingSAME*
strides
20
.gcn_attention_30/core_gcn_60/conv1d_243/conv1d?
6gcn_attention_30/core_gcn_60/conv1d_243/conv1d/SqueezeSqueeze7gcn_attention_30/core_gcn_60/conv1d_243/conv1d:output:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims

?????????28
6gcn_attention_30/core_gcn_60/conv1d_243/conv1d/Squeeze?
>gcn_attention_30/core_gcn_60/conv1d_243/BiasAdd/ReadVariableOpReadVariableOpGgcn_attention_30_core_gcn_60_conv1d_243_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_30/core_gcn_60/conv1d_243/BiasAdd/ReadVariableOp?
/gcn_attention_30/core_gcn_60/conv1d_243/BiasAddBiasAdd?gcn_attention_30/core_gcn_60/conv1d_243/conv1d/Squeeze:output:0Fgcn_attention_30/core_gcn_60/conv1d_243/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????21
/gcn_attention_30/core_gcn_60/conv1d_243/BiasAdd?
Hgcn_attention_30/core_gcn_60/batch_normalization_487/Cast/ReadVariableOpReadVariableOpQgcn_attention_30_core_gcn_60_batch_normalization_487_cast_readvariableop_resource*
_output_shapes
:*
dtype02J
Hgcn_attention_30/core_gcn_60/batch_normalization_487/Cast/ReadVariableOp?
Jgcn_attention_30/core_gcn_60/batch_normalization_487/Cast_1/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_60_batch_normalization_487_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_60/batch_normalization_487/Cast_1/ReadVariableOp?
Jgcn_attention_30/core_gcn_60/batch_normalization_487/Cast_2/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_60_batch_normalization_487_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_60/batch_normalization_487/Cast_2/ReadVariableOp?
Jgcn_attention_30/core_gcn_60/batch_normalization_487/Cast_3/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_60_batch_normalization_487_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_60/batch_normalization_487/Cast_3/ReadVariableOp?
Dgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2F
Dgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/add/y?
Bgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/addAddV2Rgcn_attention_30/core_gcn_60/batch_normalization_487/Cast_1/ReadVariableOp:value:0Mgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/add/y:output:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/add?
Dgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/RsqrtRsqrtFgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/add:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/Rsqrt?
Bgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/mulMulHgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/Rsqrt:y:0Rgcn_attention_30/core_gcn_60/batch_normalization_487/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/mul?
Dgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/mul_1Mul8gcn_attention_30/core_gcn_60/conv1d_243/BiasAdd:output:0Fgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2F
Dgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/mul_1?
Dgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/mul_2MulPgcn_attention_30/core_gcn_60/batch_normalization_487/Cast/ReadVariableOp:value:0Fgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/mul:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/mul_2?
Bgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/subSubRgcn_attention_30/core_gcn_60/batch_normalization_487/Cast_2/ReadVariableOp:value:0Hgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/sub?
Dgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/add_1AddV2Hgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/mul_1:z:0Fgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2F
Dgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/add_1?
.gcn_attention_30/core_gcn_60/activation_4/ReluReluHgcn_attention_30/core_gcn_60/batch_normalization_487/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????20
.gcn_attention_30/core_gcn_60/activation_4/Relu?
1gcn_attention_30/core_gcn_60/dropout_367/IdentityIdentity<gcn_attention_30/core_gcn_60/activation_4/Relu:activations:0*
T0*4
_output_shapes"
 :??????????????????23
1gcn_attention_30/core_gcn_60/dropout_367/Identity?
-gcn_attention_30/core_gcn_60/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-gcn_attention_30/core_gcn_60/ExpandDims_2/dim?
)gcn_attention_30/core_gcn_60/ExpandDims_2
ExpandDims:gcn_attention_30/core_gcn_60/dropout_367/Identity:output:06gcn_attention_30/core_gcn_60/ExpandDims_2/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2+
)gcn_attention_30/core_gcn_60/ExpandDims_2?
)gcn_attention_30/core_gcn_60/multiply/mulMul5gcn_attention_30/core_gcn_60/activation_2/Sigmoid:y:02gcn_attention_30/core_gcn_60/ExpandDims_2:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2+
)gcn_attention_30/core_gcn_60/multiply/mul?
9gcn_attention_30/core_gcn_60/lambda/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9gcn_attention_30/core_gcn_60/lambda/Sum/reduction_indices?
'gcn_attention_30/core_gcn_60/lambda/SumSum-gcn_attention_30/core_gcn_60/multiply/mul:z:0Bgcn_attention_30/core_gcn_60/lambda/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????2)
'gcn_attention_30/core_gcn_60/lambda/Sum?
;gcn_attention_30/core_gcn_60/lambda/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;gcn_attention_30/core_gcn_60/lambda/Sum_1/reduction_indices?
)gcn_attention_30/core_gcn_60/lambda/Sum_1Sum5gcn_attention_30/core_gcn_60/activation_2/Sigmoid:y:0Dgcn_attention_30/core_gcn_60/lambda/Sum_1/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????2+
)gcn_attention_30/core_gcn_60/lambda/Sum_1?
+gcn_attention_30/core_gcn_60/lambda_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?<2-
+gcn_attention_30/core_gcn_60/lambda_1/add/y?
)gcn_attention_30/core_gcn_60/lambda_1/addAddV22gcn_attention_30/core_gcn_60/lambda/Sum_1:output:04gcn_attention_30/core_gcn_60/lambda_1/add/y:output:0*
T0*4
_output_shapes"
 :??????????????????2+
)gcn_attention_30/core_gcn_60/lambda_1/add?
-gcn_attention_30/core_gcn_60/lambda_1/truedivRealDiv0gcn_attention_30/core_gcn_60/lambda/Sum:output:0-gcn_attention_30/core_gcn_60/lambda_1/add:z:0*
T0*4
_output_shapes"
 :??????????????????2/
-gcn_attention_30/core_gcn_60/lambda_1/truediv?
&gcn_attention_30/core_gcn_60/add_1/addAddV2:gcn_attention_30/core_gcn_60/dropout_366/Identity:output:01gcn_attention_30/core_gcn_60/lambda_1/truediv:z:0*
T0*4
_output_shapes"
 :??????????????????2(
&gcn_attention_30/core_gcn_60/add_1/add?
Cgcn_attention_30/core_gcn_60/batch_normalization_488/ReadVariableOpReadVariableOpLgcn_attention_30_core_gcn_60_batch_normalization_488_readvariableop_resource*
_output_shapes
:*
dtype02E
Cgcn_attention_30/core_gcn_60/batch_normalization_488/ReadVariableOp?
Egcn_attention_30/core_gcn_60/batch_normalization_488/ReadVariableOp_1ReadVariableOpNgcn_attention_30_core_gcn_60_batch_normalization_488_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Egcn_attention_30/core_gcn_60/batch_normalization_488/ReadVariableOp_1?
Tgcn_attention_30/core_gcn_60/batch_normalization_488/FusedBatchNormV3/ReadVariableOpReadVariableOp]gcn_attention_30_core_gcn_60_batch_normalization_488_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02V
Tgcn_attention_30/core_gcn_60/batch_normalization_488/FusedBatchNormV3/ReadVariableOp?
Vgcn_attention_30/core_gcn_60/batch_normalization_488/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_gcn_attention_30_core_gcn_60_batch_normalization_488_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02X
Vgcn_attention_30/core_gcn_60/batch_normalization_488/FusedBatchNormV3/ReadVariableOp_1?
Egcn_attention_30/core_gcn_60/batch_normalization_488/FusedBatchNormV3FusedBatchNormV3*gcn_attention_30/core_gcn_60/add/add_1:z:0Kgcn_attention_30/core_gcn_60/batch_normalization_488/ReadVariableOp:value:0Mgcn_attention_30/core_gcn_60/batch_normalization_488/ReadVariableOp_1:value:0\gcn_attention_30/core_gcn_60/batch_normalization_488/FusedBatchNormV3/ReadVariableOp:value:0^gcn_attention_30/core_gcn_60/batch_normalization_488/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2G
Egcn_attention_30/core_gcn_60/batch_normalization_488/FusedBatchNormV3?
Hgcn_attention_30/core_gcn_60/batch_normalization_489/Cast/ReadVariableOpReadVariableOpQgcn_attention_30_core_gcn_60_batch_normalization_489_cast_readvariableop_resource*
_output_shapes
:*
dtype02J
Hgcn_attention_30/core_gcn_60/batch_normalization_489/Cast/ReadVariableOp?
Jgcn_attention_30/core_gcn_60/batch_normalization_489/Cast_1/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_60_batch_normalization_489_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_60/batch_normalization_489/Cast_1/ReadVariableOp?
Jgcn_attention_30/core_gcn_60/batch_normalization_489/Cast_2/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_60_batch_normalization_489_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_60/batch_normalization_489/Cast_2/ReadVariableOp?
Jgcn_attention_30/core_gcn_60/batch_normalization_489/Cast_3/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_60_batch_normalization_489_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_60/batch_normalization_489/Cast_3/ReadVariableOp?
Dgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2F
Dgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/add/y?
Bgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/addAddV2Rgcn_attention_30/core_gcn_60/batch_normalization_489/Cast_1/ReadVariableOp:value:0Mgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/add/y:output:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/add?
Dgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/RsqrtRsqrtFgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/add:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/Rsqrt?
Bgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/mulMulHgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/Rsqrt:y:0Rgcn_attention_30/core_gcn_60/batch_normalization_489/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/mul?
Dgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/mul_1Mul*gcn_attention_30/core_gcn_60/add_1/add:z:0Fgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2F
Dgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/mul_1?
Dgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/mul_2MulPgcn_attention_30/core_gcn_60/batch_normalization_489/Cast/ReadVariableOp:value:0Fgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/mul:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/mul_2?
Bgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/subSubRgcn_attention_30/core_gcn_60/batch_normalization_489/Cast_2/ReadVariableOp:value:0Hgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/sub?
Dgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/add_1AddV2Hgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/mul_1:z:0Fgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2F
Dgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/add_1?
.gcn_attention_30/core_gcn_60/activation_5/ReluReluIgcn_attention_30/core_gcn_60/batch_normalization_488/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????20
.gcn_attention_30/core_gcn_60/activation_5/Relu?
.gcn_attention_30/core_gcn_60/activation_6/ReluReluHgcn_attention_30/core_gcn_60/batch_normalization_489/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????20
.gcn_attention_30/core_gcn_60/activation_6/Relu?
&gcn_attention_30/core_gcn_60/add_2/addAddV2.gcn_attention_30/dropout_360/Identity:output:0<gcn_attention_30/core_gcn_60/activation_6/Relu:activations:0*
T0*4
_output_shapes"
 :??????????????????2(
&gcn_attention_30/core_gcn_60/add_2/add?
&gcn_attention_30/core_gcn_60/add_3/addAddV2,gcn_attention_30/concatenate/concat:output:0<gcn_attention_30/core_gcn_60/activation_5/Relu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????2(
&gcn_attention_30/core_gcn_60/add_3/add?
<gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2>
<gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/dilation_rate?
;gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2=
;gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/filter_shape?
4gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            26
4gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/stack?
4gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/ShapeShape*gcn_attention_30/core_gcn_60/add_3/add:z:0*
T0*
_output_shapes
:26
4gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/Shape?
Bgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2D
Bgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice/stack?
Dgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice/stack_1?
Dgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice/stack_2?
<gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_sliceStridedSlice=gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/Shape:output:0Kgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice/stack:output:0Mgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice/stack_1:output:0Mgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice?
Dgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2F
Dgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_1/stack?
Fgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_1/stack_1?
Fgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_1/stack_2?
>gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_1StridedSlice=gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/Shape:output:0Mgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_1/stack:output:0Ogcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_1/stack_1:output:0Ogcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2@
>gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_1?
6gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/stack_1PackEgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice:output:0Ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_1:output:0*
N*
T0*
_output_shapes
:28
6gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/stack_1?
cgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2e
cgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice/stack?
egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2g
egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice/stack_1?
egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2g
egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice/stack_2?
]gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_sliceStridedSlice=gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/stack:output:0lgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice/stack:output:0ngcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice/stack_1:output:0ngcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2_
]gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice?
egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2g
egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_1/stack?
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2i
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_1/stack_1?
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2i
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_1/stack_2?
_gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_1StridedSlice=gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/stack:output:0ngcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_1/stack:output:0pgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_1/stack_1:output:0pgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2a
_gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_1?
Sgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/addAddV2?gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/stack_1:output:0fgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice:output:0*
T0*
_output_shapes
:2U
Sgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/add?
Ugcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/add_1AddV2Wgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/add:z:0hgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_1:output:0*
T0*
_output_shapes
:2W
Ugcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/add_1?
Sgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/modFloorModYgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/add_1:z:0Egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/dilation_rate:output:0*
T0*
_output_shapes
:2U
Sgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/mod?
Sgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/subSubEgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/dilation_rate:output:0Wgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/mod:z:0*
T0*
_output_shapes
:2U
Sgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/sub?
Ugcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/mod_1FloorModWgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/sub:z:0Egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/dilation_rate:output:0*
T0*
_output_shapes
:2W
Ugcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/mod_1?
Ugcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/add_2AddV2hgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_1:output:0Ygcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/mod_1:z:0*
T0*
_output_shapes
:2W
Ugcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/add_2?
egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_2/stack?
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_2/stack_1?
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_2/stack_2?
_gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_2StridedSlicefgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice:output:0ngcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_2/stack:output:0pgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_2/stack_1:output:0pgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_2?
egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_3/stack?
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_3/stack_1?
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_3/stack_2?
_gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_3StridedSliceYgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/add_2:z:0ngcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_3/stack:output:0pgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_3/stack_1:output:0pgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_3?
egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2g
egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_4/stack?
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_4/stack_1?
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_4/stack_2?
_gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_4StridedSlicefgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice:output:0ngcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_4/stack:output:0pgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_4/stack_1:output:0pgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_4?
egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2g
egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_5/stack?
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_5/stack_1?
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_5/stack_2?
_gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_5StridedSliceYgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/add_2:z:0ngcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_5/stack:output:0pgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_5/stack_1:output:0pgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_5?
Zgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/paddings/0Packhgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_2:output:0hgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2\
Zgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/paddings/0?
Zgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/paddings/1Packhgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_4:output:0hgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_5:output:0*
N*
T0*
_output_shapes
:2\
Zgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/paddings/1?
Xgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/paddingsPackcgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/paddings/0:output:0cgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/paddings/1:output:0*
N*
T0*
_output_shapes

:2Z
Xgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/paddings?
egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_6/stack?
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_6/stack_1?
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_6/stack_2?
_gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_6StridedSliceYgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/mod_1:z:0ngcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_6/stack:output:0pgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_6/stack_1:output:0pgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_6?
egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2g
egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_7/stack?
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_7/stack_1?
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_7/stack_2?
_gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_7StridedSliceYgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/mod_1:z:0ngcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_7/stack:output:0pgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_7/stack_1:output:0pgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_7?
Ygcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/crops/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2[
Ygcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/crops/0/0?
Wgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/crops/0Packbgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/crops/0/0:output:0hgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_6:output:0*
N*
T0*
_output_shapes
:2Y
Wgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/crops/0?
Ygcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/crops/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2[
Ygcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/crops/1/0?
Wgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/crops/1Packbgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/crops/1/0:output:0hgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/strided_slice_7:output:0*
N*
T0*
_output_shapes
:2Y
Wgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/crops/1?
Ugcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/cropsPack`gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/crops/0:output:0`gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/crops/1:output:0*
N*
T0*
_output_shapes

:2W
Ugcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/crops?
Dgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_2/stack?
Fgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_2/stack_1?
Fgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_2/stack_2?
>gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_2StridedSliceagcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/paddings:output:0Mgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_2/stack:output:0Ogcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_2/stack_1:output:0Ogcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:2@
>gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_2?
@gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 2B
@gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/concat/concat_dim?
<gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/concat/concatIdentityGgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_2:output:0*
T0*
_output_shapes

:2>
<gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/concat/concat?
Dgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_3/stack?
Fgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_3/stack_1?
Fgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_3/stack_2?
>gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_3StridedSlice^gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/required_space_to_batch_paddings/crops:output:0Mgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_3/stack:output:0Ogcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_3/stack_1:output:0Ogcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:2@
>gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_3?
Bgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/concat_1/concat_dim?
>gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/concat_1/concatIdentityGgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/strided_slice_3:output:0*
T0*
_output_shapes

:2@
>gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/concat_1/concat?
Igcn_attention_30/core_gcn_61/conv2d_123/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2K
Igcn_attention_30/core_gcn_61/conv2d_123/Conv2D/SpaceToBatchND/block_shape?
=gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/SpaceToBatchNDSpaceToBatchND*gcn_attention_30/core_gcn_60/add_3/add:z:0Rgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/SpaceToBatchND/block_shape:output:0Egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/concat/concat:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2?
=gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/SpaceToBatchND?
=gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/ReadVariableOpReadVariableOpFgcn_attention_30_core_gcn_61_conv2d_123_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02?
=gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/ReadVariableOp?
.gcn_attention_30/core_gcn_61/conv2d_123/Conv2DConv2DFgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/SpaceToBatchND:output:0Egcn_attention_30/core_gcn_61/conv2d_123/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
20
.gcn_attention_30/core_gcn_61/conv2d_123/Conv2D?
Igcn_attention_30/core_gcn_61/conv2d_123/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2K
Igcn_attention_30/core_gcn_61/conv2d_123/Conv2D/BatchToSpaceND/block_shape?
=gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/BatchToSpaceNDBatchToSpaceND7gcn_attention_30/core_gcn_61/conv2d_123/Conv2D:output:0Rgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/BatchToSpaceND/block_shape:output:0Ggcn_attention_30/core_gcn_61/conv2d_123/Conv2D/concat_1/concat:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2?
=gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/BatchToSpaceND?
>gcn_attention_30/core_gcn_61/conv2d_123/BiasAdd/ReadVariableOpReadVariableOpGgcn_attention_30_core_gcn_61_conv2d_123_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_30/core_gcn_61/conv2d_123/BiasAdd/ReadVariableOp?
/gcn_attention_30/core_gcn_61/conv2d_123/BiasAddBiasAddFgcn_attention_30/core_gcn_61/conv2d_123/Conv2D/BatchToSpaceND:output:0Fgcn_attention_30/core_gcn_61/conv2d_123/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????21
/gcn_attention_30/core_gcn_61/conv2d_123/BiasAdd?
.gcn_attention_30/core_gcn_61/activation_7/ReluRelu8gcn_attention_30/core_gcn_61/conv2d_123/BiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????20
.gcn_attention_30/core_gcn_61/activation_7/Relu?
Cgcn_attention_30/core_gcn_61/batch_normalization_490/ReadVariableOpReadVariableOpLgcn_attention_30_core_gcn_61_batch_normalization_490_readvariableop_resource*
_output_shapes
:*
dtype02E
Cgcn_attention_30/core_gcn_61/batch_normalization_490/ReadVariableOp?
Egcn_attention_30/core_gcn_61/batch_normalization_490/ReadVariableOp_1ReadVariableOpNgcn_attention_30_core_gcn_61_batch_normalization_490_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Egcn_attention_30/core_gcn_61/batch_normalization_490/ReadVariableOp_1?
Tgcn_attention_30/core_gcn_61/batch_normalization_490/FusedBatchNormV3/ReadVariableOpReadVariableOp]gcn_attention_30_core_gcn_61_batch_normalization_490_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02V
Tgcn_attention_30/core_gcn_61/batch_normalization_490/FusedBatchNormV3/ReadVariableOp?
Vgcn_attention_30/core_gcn_61/batch_normalization_490/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_gcn_attention_30_core_gcn_61_batch_normalization_490_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02X
Vgcn_attention_30/core_gcn_61/batch_normalization_490/FusedBatchNormV3/ReadVariableOp_1?
Egcn_attention_30/core_gcn_61/batch_normalization_490/FusedBatchNormV3FusedBatchNormV3<gcn_attention_30/core_gcn_61/activation_7/Relu:activations:0Kgcn_attention_30/core_gcn_61/batch_normalization_490/ReadVariableOp:value:0Mgcn_attention_30/core_gcn_61/batch_normalization_490/ReadVariableOp_1:value:0\gcn_attention_30/core_gcn_61/batch_normalization_490/FusedBatchNormV3/ReadVariableOp:value:0^gcn_attention_30/core_gcn_61/batch_normalization_490/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2G
Egcn_attention_30/core_gcn_61/batch_normalization_490/FusedBatchNormV3?
1gcn_attention_30/core_gcn_61/dropout_368/IdentityIdentityIgcn_attention_30/core_gcn_61/batch_normalization_490/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????23
1gcn_attention_30/core_gcn_61/dropout_368/Identity?
<gcn_attention_30/core_gcn_61/conv1d_244/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:2>
<gcn_attention_30/core_gcn_61/conv1d_244/conv1d/dilation_rate?
;gcn_attention_30/core_gcn_61/conv1d_244/conv1d/filter_shapeConst*
_output_shapes
:*
dtype0*!
valueB"         2=
;gcn_attention_30/core_gcn_61/conv1d_244/conv1d/filter_shape?
4gcn_attention_30/core_gcn_61/conv1d_244/conv1d/stackConst*
_output_shapes

:*
dtype0*!
valueB"      26
4gcn_attention_30/core_gcn_61/conv1d_244/conv1d/stack?
4gcn_attention_30/core_gcn_61/conv1d_244/conv1d/ShapeShape*gcn_attention_30/core_gcn_60/add_2/add:z:0*
T0*
_output_shapes
:26
4gcn_attention_30/core_gcn_61/conv1d_244/conv1d/Shape?
Bgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2D
Bgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice/stack?
Dgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice/stack_1?
Dgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice/stack_2?
<gcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_sliceStridedSlice=gcn_attention_30/core_gcn_61/conv1d_244/conv1d/Shape:output:0Kgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice/stack:output:0Mgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice/stack_1:output:0Mgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<gcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice?
6gcn_attention_30/core_gcn_61/conv1d_244/conv1d/stack_1PackEgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice:output:0*
N*
T0*
_output_shapes
:28
6gcn_attention_30/core_gcn_61/conv1d_244/conv1d/stack_1?
cgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2e
cgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice/stack?
egcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2g
egcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice/stack_1?
egcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2g
egcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice/stack_2?
]gcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_sliceStridedSlice=gcn_attention_30/core_gcn_61/conv1d_244/conv1d/stack:output:0lgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice/stack:output:0ngcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice/stack_1:output:0ngcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2_
]gcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice?
egcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2g
egcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_1/stack?
ggcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2i
ggcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1?
ggcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2i
ggcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2?
_gcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_1StridedSlice=gcn_attention_30/core_gcn_61/conv1d_244/conv1d/stack:output:0ngcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_1/stack:output:0pgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1:output:0pgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2a
_gcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_1?
Sgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/addAddV2?gcn_attention_30/core_gcn_61/conv1d_244/conv1d/stack_1:output:0fgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice:output:0*
T0*
_output_shapes
:2U
Sgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/add?
Ugcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/add_1AddV2Wgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/add:z:0hgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_1:output:0*
T0*
_output_shapes
:2W
Ugcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/add_1?
Sgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/modFloorModYgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/add_1:z:0Egcn_attention_30/core_gcn_61/conv1d_244/conv1d/dilation_rate:output:0*
T0*
_output_shapes
:2U
Sgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/mod?
Sgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/subSubEgcn_attention_30/core_gcn_61/conv1d_244/conv1d/dilation_rate:output:0Wgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/mod:z:0*
T0*
_output_shapes
:2U
Sgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/sub?
Ugcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/mod_1FloorModWgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/sub:z:0Egcn_attention_30/core_gcn_61/conv1d_244/conv1d/dilation_rate:output:0*
T0*
_output_shapes
:2W
Ugcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/mod_1?
Ugcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/add_2AddV2hgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_1:output:0Ygcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/mod_1:z:0*
T0*
_output_shapes
:2W
Ugcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/add_2?
egcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_2/stack?
ggcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1?
ggcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2?
_gcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_2StridedSlicefgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice:output:0ngcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_2/stack:output:0pgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1:output:0pgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_2?
egcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_3/stack?
ggcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1?
ggcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2?
_gcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_3StridedSliceYgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/add_2:z:0ngcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_3/stack:output:0pgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1:output:0pgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_3?
Zgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/paddings/0Packhgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_2:output:0hgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2\
Zgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/paddings/0?
Xgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/paddingsPackcgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/paddings/0:output:0*
N*
T0*
_output_shapes

:2Z
Xgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/paddings?
egcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_4/stack?
ggcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1?
ggcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2?
_gcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_4StridedSliceYgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/mod_1:z:0ngcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_4/stack:output:0pgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1:output:0pgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_4?
Ygcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/crops/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2[
Ygcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/crops/0/0?
Wgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/crops/0Packbgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/crops/0/0:output:0hgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/strided_slice_4:output:0*
N*
T0*
_output_shapes
:2Y
Wgcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/crops/0?
Ugcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/cropsPack`gcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/crops/0:output:0*
N*
T0*
_output_shapes

:2W
Ugcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/crops?
Dgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_1/stack?
Fgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_1/stack_1?
Fgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_1/stack_2?
>gcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_1StridedSliceagcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/paddings:output:0Mgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_1/stack:output:0Ogcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_1/stack_1:output:0Ogcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:2@
>gcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_1?
@gcn_attention_30/core_gcn_61/conv1d_244/conv1d/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 2B
@gcn_attention_30/core_gcn_61/conv1d_244/conv1d/concat/concat_dim?
<gcn_attention_30/core_gcn_61/conv1d_244/conv1d/concat/concatIdentityGgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_1:output:0*
T0*
_output_shapes

:2>
<gcn_attention_30/core_gcn_61/conv1d_244/conv1d/concat/concat?
Dgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_2/stack?
Fgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_2/stack_1?
Fgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_2/stack_2?
>gcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_2StridedSlice^gcn_attention_30/core_gcn_61/conv1d_244/conv1d/required_space_to_batch_paddings/crops:output:0Mgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_2/stack:output:0Ogcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_2/stack_1:output:0Ogcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:2@
>gcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_2?
Bgcn_attention_30/core_gcn_61/conv1d_244/conv1d/concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bgcn_attention_30/core_gcn_61/conv1d_244/conv1d/concat_1/concat_dim?
>gcn_attention_30/core_gcn_61/conv1d_244/conv1d/concat_1/concatIdentityGgcn_attention_30/core_gcn_61/conv1d_244/conv1d/strided_slice_2:output:0*
T0*
_output_shapes

:2@
>gcn_attention_30/core_gcn_61/conv1d_244/conv1d/concat_1/concat?
Igcn_attention_30/core_gcn_61/conv1d_244/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2K
Igcn_attention_30/core_gcn_61/conv1d_244/conv1d/SpaceToBatchND/block_shape?
=gcn_attention_30/core_gcn_61/conv1d_244/conv1d/SpaceToBatchNDSpaceToBatchND*gcn_attention_30/core_gcn_60/add_2/add:z:0Rgcn_attention_30/core_gcn_61/conv1d_244/conv1d/SpaceToBatchND/block_shape:output:0Egcn_attention_30/core_gcn_61/conv1d_244/conv1d/concat/concat:output:0*
T0*4
_output_shapes"
 :??????????????????2?
=gcn_attention_30/core_gcn_61/conv1d_244/conv1d/SpaceToBatchND?
=gcn_attention_30/core_gcn_61/conv1d_244/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2?
=gcn_attention_30/core_gcn_61/conv1d_244/conv1d/ExpandDims/dim?
9gcn_attention_30/core_gcn_61/conv1d_244/conv1d/ExpandDims
ExpandDimsFgcn_attention_30/core_gcn_61/conv1d_244/conv1d/SpaceToBatchND:output:0Fgcn_attention_30/core_gcn_61/conv1d_244/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2;
9gcn_attention_30/core_gcn_61/conv1d_244/conv1d/ExpandDims?
Jgcn_attention_30/core_gcn_61/conv1d_244/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_61_conv1d_244_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_61/conv1d_244/conv1d/ExpandDims_1/ReadVariableOp?
?gcn_attention_30/core_gcn_61/conv1d_244/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2A
?gcn_attention_30/core_gcn_61/conv1d_244/conv1d/ExpandDims_1/dim?
;gcn_attention_30/core_gcn_61/conv1d_244/conv1d/ExpandDims_1
ExpandDimsRgcn_attention_30/core_gcn_61/conv1d_244/conv1d/ExpandDims_1/ReadVariableOp:value:0Hgcn_attention_30/core_gcn_61/conv1d_244/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2=
;gcn_attention_30/core_gcn_61/conv1d_244/conv1d/ExpandDims_1?
.gcn_attention_30/core_gcn_61/conv1d_244/conv1dConv2DBgcn_attention_30/core_gcn_61/conv1d_244/conv1d/ExpandDims:output:0Dgcn_attention_30/core_gcn_61/conv1d_244/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingVALID*
strides
20
.gcn_attention_30/core_gcn_61/conv1d_244/conv1d?
6gcn_attention_30/core_gcn_61/conv1d_244/conv1d/SqueezeSqueeze7gcn_attention_30/core_gcn_61/conv1d_244/conv1d:output:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims

?????????28
6gcn_attention_30/core_gcn_61/conv1d_244/conv1d/Squeeze?
Igcn_attention_30/core_gcn_61/conv1d_244/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2K
Igcn_attention_30/core_gcn_61/conv1d_244/conv1d/BatchToSpaceND/block_shape?
=gcn_attention_30/core_gcn_61/conv1d_244/conv1d/BatchToSpaceNDBatchToSpaceND?gcn_attention_30/core_gcn_61/conv1d_244/conv1d/Squeeze:output:0Rgcn_attention_30/core_gcn_61/conv1d_244/conv1d/BatchToSpaceND/block_shape:output:0Ggcn_attention_30/core_gcn_61/conv1d_244/conv1d/concat_1/concat:output:0*
T0*4
_output_shapes"
 :??????????????????2?
=gcn_attention_30/core_gcn_61/conv1d_244/conv1d/BatchToSpaceND?
>gcn_attention_30/core_gcn_61/conv1d_244/BiasAdd/ReadVariableOpReadVariableOpGgcn_attention_30_core_gcn_61_conv1d_244_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_30/core_gcn_61/conv1d_244/BiasAdd/ReadVariableOp?
/gcn_attention_30/core_gcn_61/conv1d_244/BiasAddBiasAddFgcn_attention_30/core_gcn_61/conv1d_244/conv1d/BatchToSpaceND:output:0Fgcn_attention_30/core_gcn_61/conv1d_244/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????21
/gcn_attention_30/core_gcn_61/conv1d_244/BiasAdd?
.gcn_attention_30/core_gcn_61/activation_8/ReluRelu8gcn_attention_30/core_gcn_61/conv1d_244/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????20
.gcn_attention_30/core_gcn_61/activation_8/Relu?
Hgcn_attention_30/core_gcn_61/batch_normalization_491/Cast/ReadVariableOpReadVariableOpQgcn_attention_30_core_gcn_61_batch_normalization_491_cast_readvariableop_resource*
_output_shapes
:*
dtype02J
Hgcn_attention_30/core_gcn_61/batch_normalization_491/Cast/ReadVariableOp?
Jgcn_attention_30/core_gcn_61/batch_normalization_491/Cast_1/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_61_batch_normalization_491_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_61/batch_normalization_491/Cast_1/ReadVariableOp?
Jgcn_attention_30/core_gcn_61/batch_normalization_491/Cast_2/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_61_batch_normalization_491_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_61/batch_normalization_491/Cast_2/ReadVariableOp?
Jgcn_attention_30/core_gcn_61/batch_normalization_491/Cast_3/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_61_batch_normalization_491_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_61/batch_normalization_491/Cast_3/ReadVariableOp?
Dgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2F
Dgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/add/y?
Bgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/addAddV2Rgcn_attention_30/core_gcn_61/batch_normalization_491/Cast_1/ReadVariableOp:value:0Mgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/add/y:output:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/add?
Dgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/RsqrtRsqrtFgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/add:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/Rsqrt?
Bgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/mulMulHgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/Rsqrt:y:0Rgcn_attention_30/core_gcn_61/batch_normalization_491/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/mul?
Dgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/mul_1Mul<gcn_attention_30/core_gcn_61/activation_8/Relu:activations:0Fgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2F
Dgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/mul_1?
Dgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/mul_2MulPgcn_attention_30/core_gcn_61/batch_normalization_491/Cast/ReadVariableOp:value:0Fgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/mul:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/mul_2?
Bgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/subSubRgcn_attention_30/core_gcn_61/batch_normalization_491/Cast_2/ReadVariableOp:value:0Hgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/sub?
Dgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/add_1AddV2Hgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/mul_1:z:0Fgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2F
Dgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/add_1?
1gcn_attention_30/core_gcn_61/dropout_369/IdentityIdentityHgcn_attention_30/core_gcn_61/batch_normalization_491/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????23
1gcn_attention_30/core_gcn_61/dropout_369/Identity?
+gcn_attention_30/core_gcn_61/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+gcn_attention_30/core_gcn_61/ExpandDims/dim?
'gcn_attention_30/core_gcn_61/ExpandDims
ExpandDims:gcn_attention_30/core_gcn_61/dropout_369/Identity:output:04gcn_attention_30/core_gcn_61/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'gcn_attention_30/core_gcn_61/ExpandDims?
-gcn_attention_30/core_gcn_61/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-gcn_attention_30/core_gcn_61/ExpandDims_1/dim?
)gcn_attention_30/core_gcn_61/ExpandDims_1
ExpandDims:gcn_attention_30/core_gcn_61/dropout_369/Identity:output:06gcn_attention_30/core_gcn_61/ExpandDims_1/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2+
)gcn_attention_30/core_gcn_61/ExpandDims_1?
&gcn_attention_30/core_gcn_61/add_4/addAddV2:gcn_attention_30/core_gcn_61/dropout_368/Identity:output:02gcn_attention_30/core_gcn_61/ExpandDims_1:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2(
&gcn_attention_30/core_gcn_61/add_4/add?
(gcn_attention_30/core_gcn_61/add_4/add_1AddV2*gcn_attention_30/core_gcn_61/add_4/add:z:00gcn_attention_30/core_gcn_61/ExpandDims:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2*
(gcn_attention_30/core_gcn_61/add_4/add_1?
1gcn_attention_30/core_gcn_61/activation_9/SigmoidSigmoid,gcn_attention_30/core_gcn_61/add_4/add_1:z:0*
T0*A
_output_shapes/
-:+???????????????????????????23
1gcn_attention_30/core_gcn_61/activation_9/Sigmoid?
<gcn_attention_30/core_gcn_61/conv1d_245/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:2>
<gcn_attention_30/core_gcn_61/conv1d_245/conv1d/dilation_rate?
;gcn_attention_30/core_gcn_61/conv1d_245/conv1d/filter_shapeConst*
_output_shapes
:*
dtype0*!
valueB"         2=
;gcn_attention_30/core_gcn_61/conv1d_245/conv1d/filter_shape?
4gcn_attention_30/core_gcn_61/conv1d_245/conv1d/stackConst*
_output_shapes

:*
dtype0*!
valueB"      26
4gcn_attention_30/core_gcn_61/conv1d_245/conv1d/stack?
4gcn_attention_30/core_gcn_61/conv1d_245/conv1d/ShapeShape*gcn_attention_30/core_gcn_60/add_2/add:z:0*
T0*
_output_shapes
:26
4gcn_attention_30/core_gcn_61/conv1d_245/conv1d/Shape?
Bgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2D
Bgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice/stack?
Dgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice/stack_1?
Dgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice/stack_2?
<gcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_sliceStridedSlice=gcn_attention_30/core_gcn_61/conv1d_245/conv1d/Shape:output:0Kgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice/stack:output:0Mgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice/stack_1:output:0Mgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<gcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice?
6gcn_attention_30/core_gcn_61/conv1d_245/conv1d/stack_1PackEgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice:output:0*
N*
T0*
_output_shapes
:28
6gcn_attention_30/core_gcn_61/conv1d_245/conv1d/stack_1?
cgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2e
cgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice/stack?
egcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2g
egcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice/stack_1?
egcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2g
egcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice/stack_2?
]gcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_sliceStridedSlice=gcn_attention_30/core_gcn_61/conv1d_245/conv1d/stack:output:0lgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice/stack:output:0ngcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice/stack_1:output:0ngcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2_
]gcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice?
egcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2g
egcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_1/stack?
ggcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2i
ggcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1?
ggcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2i
ggcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2?
_gcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_1StridedSlice=gcn_attention_30/core_gcn_61/conv1d_245/conv1d/stack:output:0ngcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_1/stack:output:0pgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1:output:0pgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2a
_gcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_1?
Sgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/addAddV2?gcn_attention_30/core_gcn_61/conv1d_245/conv1d/stack_1:output:0fgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice:output:0*
T0*
_output_shapes
:2U
Sgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/add?
Ugcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/add_1AddV2Wgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/add:z:0hgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_1:output:0*
T0*
_output_shapes
:2W
Ugcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/add_1?
Sgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/modFloorModYgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/add_1:z:0Egcn_attention_30/core_gcn_61/conv1d_245/conv1d/dilation_rate:output:0*
T0*
_output_shapes
:2U
Sgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/mod?
Sgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/subSubEgcn_attention_30/core_gcn_61/conv1d_245/conv1d/dilation_rate:output:0Wgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/mod:z:0*
T0*
_output_shapes
:2U
Sgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/sub?
Ugcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/mod_1FloorModWgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/sub:z:0Egcn_attention_30/core_gcn_61/conv1d_245/conv1d/dilation_rate:output:0*
T0*
_output_shapes
:2W
Ugcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/mod_1?
Ugcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/add_2AddV2hgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_1:output:0Ygcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/mod_1:z:0*
T0*
_output_shapes
:2W
Ugcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/add_2?
egcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_2/stack?
ggcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1?
ggcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2?
_gcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_2StridedSlicefgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice:output:0ngcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_2/stack:output:0pgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1:output:0pgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_2?
egcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_3/stack?
ggcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1?
ggcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2?
_gcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_3StridedSliceYgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/add_2:z:0ngcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_3/stack:output:0pgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1:output:0pgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_3?
Zgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/paddings/0Packhgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_2:output:0hgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2\
Zgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/paddings/0?
Xgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/paddingsPackcgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/paddings/0:output:0*
N*
T0*
_output_shapes

:2Z
Xgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/paddings?
egcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_4/stack?
ggcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1?
ggcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2?
_gcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_4StridedSliceYgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/mod_1:z:0ngcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_4/stack:output:0pgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1:output:0pgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_4?
Ygcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/crops/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2[
Ygcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/crops/0/0?
Wgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/crops/0Packbgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/crops/0/0:output:0hgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/strided_slice_4:output:0*
N*
T0*
_output_shapes
:2Y
Wgcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/crops/0?
Ugcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/cropsPack`gcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/crops/0:output:0*
N*
T0*
_output_shapes

:2W
Ugcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/crops?
Dgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_1/stack?
Fgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_1/stack_1?
Fgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_1/stack_2?
>gcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_1StridedSliceagcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/paddings:output:0Mgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_1/stack:output:0Ogcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_1/stack_1:output:0Ogcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:2@
>gcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_1?
@gcn_attention_30/core_gcn_61/conv1d_245/conv1d/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 2B
@gcn_attention_30/core_gcn_61/conv1d_245/conv1d/concat/concat_dim?
<gcn_attention_30/core_gcn_61/conv1d_245/conv1d/concat/concatIdentityGgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_1:output:0*
T0*
_output_shapes

:2>
<gcn_attention_30/core_gcn_61/conv1d_245/conv1d/concat/concat?
Dgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_2/stack?
Fgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_2/stack_1?
Fgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_2/stack_2?
>gcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_2StridedSlice^gcn_attention_30/core_gcn_61/conv1d_245/conv1d/required_space_to_batch_paddings/crops:output:0Mgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_2/stack:output:0Ogcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_2/stack_1:output:0Ogcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:2@
>gcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_2?
Bgcn_attention_30/core_gcn_61/conv1d_245/conv1d/concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bgcn_attention_30/core_gcn_61/conv1d_245/conv1d/concat_1/concat_dim?
>gcn_attention_30/core_gcn_61/conv1d_245/conv1d/concat_1/concatIdentityGgcn_attention_30/core_gcn_61/conv1d_245/conv1d/strided_slice_2:output:0*
T0*
_output_shapes

:2@
>gcn_attention_30/core_gcn_61/conv1d_245/conv1d/concat_1/concat?
Igcn_attention_30/core_gcn_61/conv1d_245/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2K
Igcn_attention_30/core_gcn_61/conv1d_245/conv1d/SpaceToBatchND/block_shape?
=gcn_attention_30/core_gcn_61/conv1d_245/conv1d/SpaceToBatchNDSpaceToBatchND*gcn_attention_30/core_gcn_60/add_2/add:z:0Rgcn_attention_30/core_gcn_61/conv1d_245/conv1d/SpaceToBatchND/block_shape:output:0Egcn_attention_30/core_gcn_61/conv1d_245/conv1d/concat/concat:output:0*
T0*4
_output_shapes"
 :??????????????????2?
=gcn_attention_30/core_gcn_61/conv1d_245/conv1d/SpaceToBatchND?
=gcn_attention_30/core_gcn_61/conv1d_245/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2?
=gcn_attention_30/core_gcn_61/conv1d_245/conv1d/ExpandDims/dim?
9gcn_attention_30/core_gcn_61/conv1d_245/conv1d/ExpandDims
ExpandDimsFgcn_attention_30/core_gcn_61/conv1d_245/conv1d/SpaceToBatchND:output:0Fgcn_attention_30/core_gcn_61/conv1d_245/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2;
9gcn_attention_30/core_gcn_61/conv1d_245/conv1d/ExpandDims?
Jgcn_attention_30/core_gcn_61/conv1d_245/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_61_conv1d_245_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_61/conv1d_245/conv1d/ExpandDims_1/ReadVariableOp?
?gcn_attention_30/core_gcn_61/conv1d_245/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2A
?gcn_attention_30/core_gcn_61/conv1d_245/conv1d/ExpandDims_1/dim?
;gcn_attention_30/core_gcn_61/conv1d_245/conv1d/ExpandDims_1
ExpandDimsRgcn_attention_30/core_gcn_61/conv1d_245/conv1d/ExpandDims_1/ReadVariableOp:value:0Hgcn_attention_30/core_gcn_61/conv1d_245/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2=
;gcn_attention_30/core_gcn_61/conv1d_245/conv1d/ExpandDims_1?
.gcn_attention_30/core_gcn_61/conv1d_245/conv1dConv2DBgcn_attention_30/core_gcn_61/conv1d_245/conv1d/ExpandDims:output:0Dgcn_attention_30/core_gcn_61/conv1d_245/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingVALID*
strides
20
.gcn_attention_30/core_gcn_61/conv1d_245/conv1d?
6gcn_attention_30/core_gcn_61/conv1d_245/conv1d/SqueezeSqueeze7gcn_attention_30/core_gcn_61/conv1d_245/conv1d:output:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims

?????????28
6gcn_attention_30/core_gcn_61/conv1d_245/conv1d/Squeeze?
Igcn_attention_30/core_gcn_61/conv1d_245/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2K
Igcn_attention_30/core_gcn_61/conv1d_245/conv1d/BatchToSpaceND/block_shape?
=gcn_attention_30/core_gcn_61/conv1d_245/conv1d/BatchToSpaceNDBatchToSpaceND?gcn_attention_30/core_gcn_61/conv1d_245/conv1d/Squeeze:output:0Rgcn_attention_30/core_gcn_61/conv1d_245/conv1d/BatchToSpaceND/block_shape:output:0Ggcn_attention_30/core_gcn_61/conv1d_245/conv1d/concat_1/concat:output:0*
T0*4
_output_shapes"
 :??????????????????2?
=gcn_attention_30/core_gcn_61/conv1d_245/conv1d/BatchToSpaceND?
>gcn_attention_30/core_gcn_61/conv1d_245/BiasAdd/ReadVariableOpReadVariableOpGgcn_attention_30_core_gcn_61_conv1d_245_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_30/core_gcn_61/conv1d_245/BiasAdd/ReadVariableOp?
/gcn_attention_30/core_gcn_61/conv1d_245/BiasAddBiasAddFgcn_attention_30/core_gcn_61/conv1d_245/conv1d/BatchToSpaceND:output:0Fgcn_attention_30/core_gcn_61/conv1d_245/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????21
/gcn_attention_30/core_gcn_61/conv1d_245/BiasAdd?
Hgcn_attention_30/core_gcn_61/batch_normalization_492/Cast/ReadVariableOpReadVariableOpQgcn_attention_30_core_gcn_61_batch_normalization_492_cast_readvariableop_resource*
_output_shapes
:*
dtype02J
Hgcn_attention_30/core_gcn_61/batch_normalization_492/Cast/ReadVariableOp?
Jgcn_attention_30/core_gcn_61/batch_normalization_492/Cast_1/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_61_batch_normalization_492_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_61/batch_normalization_492/Cast_1/ReadVariableOp?
Jgcn_attention_30/core_gcn_61/batch_normalization_492/Cast_2/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_61_batch_normalization_492_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_61/batch_normalization_492/Cast_2/ReadVariableOp?
Jgcn_attention_30/core_gcn_61/batch_normalization_492/Cast_3/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_61_batch_normalization_492_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_61/batch_normalization_492/Cast_3/ReadVariableOp?
Dgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2F
Dgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/add/y?
Bgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/addAddV2Rgcn_attention_30/core_gcn_61/batch_normalization_492/Cast_1/ReadVariableOp:value:0Mgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/add/y:output:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/add?
Dgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/RsqrtRsqrtFgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/add:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/Rsqrt?
Bgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/mulMulHgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/Rsqrt:y:0Rgcn_attention_30/core_gcn_61/batch_normalization_492/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/mul?
Dgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/mul_1Mul8gcn_attention_30/core_gcn_61/conv1d_245/BiasAdd:output:0Fgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2F
Dgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/mul_1?
Dgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/mul_2MulPgcn_attention_30/core_gcn_61/batch_normalization_492/Cast/ReadVariableOp:value:0Fgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/mul:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/mul_2?
Bgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/subSubRgcn_attention_30/core_gcn_61/batch_normalization_492/Cast_2/ReadVariableOp:value:0Hgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/sub?
Dgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/add_1AddV2Hgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/mul_1:z:0Fgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2F
Dgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/add_1?
/gcn_attention_30/core_gcn_61/activation_10/ReluReluHgcn_attention_30/core_gcn_61/batch_normalization_492/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????21
/gcn_attention_30/core_gcn_61/activation_10/Relu?
1gcn_attention_30/core_gcn_61/dropout_370/IdentityIdentity=gcn_attention_30/core_gcn_61/activation_10/Relu:activations:0*
T0*4
_output_shapes"
 :??????????????????23
1gcn_attention_30/core_gcn_61/dropout_370/Identity?
<gcn_attention_30/core_gcn_61/conv1d_246/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:2>
<gcn_attention_30/core_gcn_61/conv1d_246/conv1d/dilation_rate?
;gcn_attention_30/core_gcn_61/conv1d_246/conv1d/filter_shapeConst*
_output_shapes
:*
dtype0*!
valueB"         2=
;gcn_attention_30/core_gcn_61/conv1d_246/conv1d/filter_shape?
4gcn_attention_30/core_gcn_61/conv1d_246/conv1d/stackConst*
_output_shapes

:*
dtype0*!
valueB"      26
4gcn_attention_30/core_gcn_61/conv1d_246/conv1d/stack?
4gcn_attention_30/core_gcn_61/conv1d_246/conv1d/ShapeShape*gcn_attention_30/core_gcn_60/add_2/add:z:0*
T0*
_output_shapes
:26
4gcn_attention_30/core_gcn_61/conv1d_246/conv1d/Shape?
Bgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2D
Bgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice/stack?
Dgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice/stack_1?
Dgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice/stack_2?
<gcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_sliceStridedSlice=gcn_attention_30/core_gcn_61/conv1d_246/conv1d/Shape:output:0Kgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice/stack:output:0Mgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice/stack_1:output:0Mgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<gcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice?
6gcn_attention_30/core_gcn_61/conv1d_246/conv1d/stack_1PackEgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice:output:0*
N*
T0*
_output_shapes
:28
6gcn_attention_30/core_gcn_61/conv1d_246/conv1d/stack_1?
cgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2e
cgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice/stack?
egcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2g
egcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice/stack_1?
egcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2g
egcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice/stack_2?
]gcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_sliceStridedSlice=gcn_attention_30/core_gcn_61/conv1d_246/conv1d/stack:output:0lgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice/stack:output:0ngcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice/stack_1:output:0ngcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2_
]gcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice?
egcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2g
egcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_1/stack?
ggcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2i
ggcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1?
ggcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2i
ggcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2?
_gcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_1StridedSlice=gcn_attention_30/core_gcn_61/conv1d_246/conv1d/stack:output:0ngcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_1/stack:output:0pgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1:output:0pgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2a
_gcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_1?
Sgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/addAddV2?gcn_attention_30/core_gcn_61/conv1d_246/conv1d/stack_1:output:0fgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice:output:0*
T0*
_output_shapes
:2U
Sgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/add?
Ugcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/add_1AddV2Wgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/add:z:0hgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_1:output:0*
T0*
_output_shapes
:2W
Ugcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/add_1?
Sgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/modFloorModYgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/add_1:z:0Egcn_attention_30/core_gcn_61/conv1d_246/conv1d/dilation_rate:output:0*
T0*
_output_shapes
:2U
Sgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/mod?
Sgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/subSubEgcn_attention_30/core_gcn_61/conv1d_246/conv1d/dilation_rate:output:0Wgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/mod:z:0*
T0*
_output_shapes
:2U
Sgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/sub?
Ugcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/mod_1FloorModWgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/sub:z:0Egcn_attention_30/core_gcn_61/conv1d_246/conv1d/dilation_rate:output:0*
T0*
_output_shapes
:2W
Ugcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/mod_1?
Ugcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/add_2AddV2hgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_1:output:0Ygcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/mod_1:z:0*
T0*
_output_shapes
:2W
Ugcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/add_2?
egcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_2/stack?
ggcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1?
ggcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2?
_gcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_2StridedSlicefgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice:output:0ngcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_2/stack:output:0pgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1:output:0pgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_2?
egcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_3/stack?
ggcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1?
ggcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2?
_gcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_3StridedSliceYgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/add_2:z:0ngcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_3/stack:output:0pgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1:output:0pgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_3?
Zgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/paddings/0Packhgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_2:output:0hgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2\
Zgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/paddings/0?
Xgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/paddingsPackcgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/paddings/0:output:0*
N*
T0*
_output_shapes

:2Z
Xgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/paddings?
egcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_4/stack?
ggcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1?
ggcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2?
_gcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_4StridedSliceYgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/mod_1:z:0ngcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_4/stack:output:0pgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1:output:0pgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_4?
Ygcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/crops/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2[
Ygcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/crops/0/0?
Wgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/crops/0Packbgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/crops/0/0:output:0hgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/strided_slice_4:output:0*
N*
T0*
_output_shapes
:2Y
Wgcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/crops/0?
Ugcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/cropsPack`gcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/crops/0:output:0*
N*
T0*
_output_shapes

:2W
Ugcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/crops?
Dgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_1/stack?
Fgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_1/stack_1?
Fgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_1/stack_2?
>gcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_1StridedSliceagcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/paddings:output:0Mgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_1/stack:output:0Ogcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_1/stack_1:output:0Ogcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:2@
>gcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_1?
@gcn_attention_30/core_gcn_61/conv1d_246/conv1d/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 2B
@gcn_attention_30/core_gcn_61/conv1d_246/conv1d/concat/concat_dim?
<gcn_attention_30/core_gcn_61/conv1d_246/conv1d/concat/concatIdentityGgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_1:output:0*
T0*
_output_shapes

:2>
<gcn_attention_30/core_gcn_61/conv1d_246/conv1d/concat/concat?
Dgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_2/stack?
Fgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_2/stack_1?
Fgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_2/stack_2?
>gcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_2StridedSlice^gcn_attention_30/core_gcn_61/conv1d_246/conv1d/required_space_to_batch_paddings/crops:output:0Mgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_2/stack:output:0Ogcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_2/stack_1:output:0Ogcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:2@
>gcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_2?
Bgcn_attention_30/core_gcn_61/conv1d_246/conv1d/concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bgcn_attention_30/core_gcn_61/conv1d_246/conv1d/concat_1/concat_dim?
>gcn_attention_30/core_gcn_61/conv1d_246/conv1d/concat_1/concatIdentityGgcn_attention_30/core_gcn_61/conv1d_246/conv1d/strided_slice_2:output:0*
T0*
_output_shapes

:2@
>gcn_attention_30/core_gcn_61/conv1d_246/conv1d/concat_1/concat?
Igcn_attention_30/core_gcn_61/conv1d_246/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2K
Igcn_attention_30/core_gcn_61/conv1d_246/conv1d/SpaceToBatchND/block_shape?
=gcn_attention_30/core_gcn_61/conv1d_246/conv1d/SpaceToBatchNDSpaceToBatchND*gcn_attention_30/core_gcn_60/add_2/add:z:0Rgcn_attention_30/core_gcn_61/conv1d_246/conv1d/SpaceToBatchND/block_shape:output:0Egcn_attention_30/core_gcn_61/conv1d_246/conv1d/concat/concat:output:0*
T0*4
_output_shapes"
 :??????????????????2?
=gcn_attention_30/core_gcn_61/conv1d_246/conv1d/SpaceToBatchND?
=gcn_attention_30/core_gcn_61/conv1d_246/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2?
=gcn_attention_30/core_gcn_61/conv1d_246/conv1d/ExpandDims/dim?
9gcn_attention_30/core_gcn_61/conv1d_246/conv1d/ExpandDims
ExpandDimsFgcn_attention_30/core_gcn_61/conv1d_246/conv1d/SpaceToBatchND:output:0Fgcn_attention_30/core_gcn_61/conv1d_246/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2;
9gcn_attention_30/core_gcn_61/conv1d_246/conv1d/ExpandDims?
Jgcn_attention_30/core_gcn_61/conv1d_246/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_61_conv1d_246_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_61/conv1d_246/conv1d/ExpandDims_1/ReadVariableOp?
?gcn_attention_30/core_gcn_61/conv1d_246/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2A
?gcn_attention_30/core_gcn_61/conv1d_246/conv1d/ExpandDims_1/dim?
;gcn_attention_30/core_gcn_61/conv1d_246/conv1d/ExpandDims_1
ExpandDimsRgcn_attention_30/core_gcn_61/conv1d_246/conv1d/ExpandDims_1/ReadVariableOp:value:0Hgcn_attention_30/core_gcn_61/conv1d_246/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2=
;gcn_attention_30/core_gcn_61/conv1d_246/conv1d/ExpandDims_1?
.gcn_attention_30/core_gcn_61/conv1d_246/conv1dConv2DBgcn_attention_30/core_gcn_61/conv1d_246/conv1d/ExpandDims:output:0Dgcn_attention_30/core_gcn_61/conv1d_246/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingVALID*
strides
20
.gcn_attention_30/core_gcn_61/conv1d_246/conv1d?
6gcn_attention_30/core_gcn_61/conv1d_246/conv1d/SqueezeSqueeze7gcn_attention_30/core_gcn_61/conv1d_246/conv1d:output:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims

?????????28
6gcn_attention_30/core_gcn_61/conv1d_246/conv1d/Squeeze?
Igcn_attention_30/core_gcn_61/conv1d_246/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2K
Igcn_attention_30/core_gcn_61/conv1d_246/conv1d/BatchToSpaceND/block_shape?
=gcn_attention_30/core_gcn_61/conv1d_246/conv1d/BatchToSpaceNDBatchToSpaceND?gcn_attention_30/core_gcn_61/conv1d_246/conv1d/Squeeze:output:0Rgcn_attention_30/core_gcn_61/conv1d_246/conv1d/BatchToSpaceND/block_shape:output:0Ggcn_attention_30/core_gcn_61/conv1d_246/conv1d/concat_1/concat:output:0*
T0*4
_output_shapes"
 :??????????????????2?
=gcn_attention_30/core_gcn_61/conv1d_246/conv1d/BatchToSpaceND?
>gcn_attention_30/core_gcn_61/conv1d_246/BiasAdd/ReadVariableOpReadVariableOpGgcn_attention_30_core_gcn_61_conv1d_246_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_30/core_gcn_61/conv1d_246/BiasAdd/ReadVariableOp?
/gcn_attention_30/core_gcn_61/conv1d_246/BiasAddBiasAddFgcn_attention_30/core_gcn_61/conv1d_246/conv1d/BatchToSpaceND:output:0Fgcn_attention_30/core_gcn_61/conv1d_246/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????21
/gcn_attention_30/core_gcn_61/conv1d_246/BiasAdd?
Hgcn_attention_30/core_gcn_61/batch_normalization_493/Cast/ReadVariableOpReadVariableOpQgcn_attention_30_core_gcn_61_batch_normalization_493_cast_readvariableop_resource*
_output_shapes
:*
dtype02J
Hgcn_attention_30/core_gcn_61/batch_normalization_493/Cast/ReadVariableOp?
Jgcn_attention_30/core_gcn_61/batch_normalization_493/Cast_1/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_61_batch_normalization_493_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_61/batch_normalization_493/Cast_1/ReadVariableOp?
Jgcn_attention_30/core_gcn_61/batch_normalization_493/Cast_2/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_61_batch_normalization_493_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_61/batch_normalization_493/Cast_2/ReadVariableOp?
Jgcn_attention_30/core_gcn_61/batch_normalization_493/Cast_3/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_61_batch_normalization_493_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_61/batch_normalization_493/Cast_3/ReadVariableOp?
Dgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2F
Dgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/add/y?
Bgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/addAddV2Rgcn_attention_30/core_gcn_61/batch_normalization_493/Cast_1/ReadVariableOp:value:0Mgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/add/y:output:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/add?
Dgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/RsqrtRsqrtFgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/add:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/Rsqrt?
Bgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/mulMulHgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/Rsqrt:y:0Rgcn_attention_30/core_gcn_61/batch_normalization_493/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/mul?
Dgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/mul_1Mul8gcn_attention_30/core_gcn_61/conv1d_246/BiasAdd:output:0Fgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2F
Dgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/mul_1?
Dgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/mul_2MulPgcn_attention_30/core_gcn_61/batch_normalization_493/Cast/ReadVariableOp:value:0Fgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/mul:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/mul_2?
Bgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/subSubRgcn_attention_30/core_gcn_61/batch_normalization_493/Cast_2/ReadVariableOp:value:0Hgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/sub?
Dgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/add_1AddV2Hgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/mul_1:z:0Fgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2F
Dgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/add_1?
/gcn_attention_30/core_gcn_61/activation_11/ReluReluHgcn_attention_30/core_gcn_61/batch_normalization_493/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????21
/gcn_attention_30/core_gcn_61/activation_11/Relu?
1gcn_attention_30/core_gcn_61/dropout_371/IdentityIdentity=gcn_attention_30/core_gcn_61/activation_11/Relu:activations:0*
T0*4
_output_shapes"
 :??????????????????23
1gcn_attention_30/core_gcn_61/dropout_371/Identity?
-gcn_attention_30/core_gcn_61/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-gcn_attention_30/core_gcn_61/ExpandDims_2/dim?
)gcn_attention_30/core_gcn_61/ExpandDims_2
ExpandDims:gcn_attention_30/core_gcn_61/dropout_371/Identity:output:06gcn_attention_30/core_gcn_61/ExpandDims_2/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2+
)gcn_attention_30/core_gcn_61/ExpandDims_2?
+gcn_attention_30/core_gcn_61/multiply_1/mulMul5gcn_attention_30/core_gcn_61/activation_9/Sigmoid:y:02gcn_attention_30/core_gcn_61/ExpandDims_2:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2-
+gcn_attention_30/core_gcn_61/multiply_1/mul?
;gcn_attention_30/core_gcn_61/lambda_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;gcn_attention_30/core_gcn_61/lambda_2/Sum/reduction_indices?
)gcn_attention_30/core_gcn_61/lambda_2/SumSum/gcn_attention_30/core_gcn_61/multiply_1/mul:z:0Dgcn_attention_30/core_gcn_61/lambda_2/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????2+
)gcn_attention_30/core_gcn_61/lambda_2/Sum?
=gcn_attention_30/core_gcn_61/lambda_2/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2?
=gcn_attention_30/core_gcn_61/lambda_2/Sum_1/reduction_indices?
+gcn_attention_30/core_gcn_61/lambda_2/Sum_1Sum5gcn_attention_30/core_gcn_61/activation_9/Sigmoid:y:0Fgcn_attention_30/core_gcn_61/lambda_2/Sum_1/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????2-
+gcn_attention_30/core_gcn_61/lambda_2/Sum_1?
+gcn_attention_30/core_gcn_61/lambda_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?<2-
+gcn_attention_30/core_gcn_61/lambda_3/add/y?
)gcn_attention_30/core_gcn_61/lambda_3/addAddV24gcn_attention_30/core_gcn_61/lambda_2/Sum_1:output:04gcn_attention_30/core_gcn_61/lambda_3/add/y:output:0*
T0*4
_output_shapes"
 :??????????????????2+
)gcn_attention_30/core_gcn_61/lambda_3/add?
-gcn_attention_30/core_gcn_61/lambda_3/truedivRealDiv2gcn_attention_30/core_gcn_61/lambda_2/Sum:output:0-gcn_attention_30/core_gcn_61/lambda_3/add:z:0*
T0*4
_output_shapes"
 :??????????????????2/
-gcn_attention_30/core_gcn_61/lambda_3/truediv?
&gcn_attention_30/core_gcn_61/add_5/addAddV2:gcn_attention_30/core_gcn_61/dropout_370/Identity:output:01gcn_attention_30/core_gcn_61/lambda_3/truediv:z:0*
T0*4
_output_shapes"
 :??????????????????2(
&gcn_attention_30/core_gcn_61/add_5/add?
Cgcn_attention_30/core_gcn_61/batch_normalization_494/ReadVariableOpReadVariableOpLgcn_attention_30_core_gcn_61_batch_normalization_494_readvariableop_resource*
_output_shapes
:*
dtype02E
Cgcn_attention_30/core_gcn_61/batch_normalization_494/ReadVariableOp?
Egcn_attention_30/core_gcn_61/batch_normalization_494/ReadVariableOp_1ReadVariableOpNgcn_attention_30_core_gcn_61_batch_normalization_494_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Egcn_attention_30/core_gcn_61/batch_normalization_494/ReadVariableOp_1?
Tgcn_attention_30/core_gcn_61/batch_normalization_494/FusedBatchNormV3/ReadVariableOpReadVariableOp]gcn_attention_30_core_gcn_61_batch_normalization_494_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02V
Tgcn_attention_30/core_gcn_61/batch_normalization_494/FusedBatchNormV3/ReadVariableOp?
Vgcn_attention_30/core_gcn_61/batch_normalization_494/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_gcn_attention_30_core_gcn_61_batch_normalization_494_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02X
Vgcn_attention_30/core_gcn_61/batch_normalization_494/FusedBatchNormV3/ReadVariableOp_1?
Egcn_attention_30/core_gcn_61/batch_normalization_494/FusedBatchNormV3FusedBatchNormV3,gcn_attention_30/core_gcn_61/add_4/add_1:z:0Kgcn_attention_30/core_gcn_61/batch_normalization_494/ReadVariableOp:value:0Mgcn_attention_30/core_gcn_61/batch_normalization_494/ReadVariableOp_1:value:0\gcn_attention_30/core_gcn_61/batch_normalization_494/FusedBatchNormV3/ReadVariableOp:value:0^gcn_attention_30/core_gcn_61/batch_normalization_494/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2G
Egcn_attention_30/core_gcn_61/batch_normalization_494/FusedBatchNormV3?
Hgcn_attention_30/core_gcn_61/batch_normalization_495/Cast/ReadVariableOpReadVariableOpQgcn_attention_30_core_gcn_61_batch_normalization_495_cast_readvariableop_resource*
_output_shapes
:*
dtype02J
Hgcn_attention_30/core_gcn_61/batch_normalization_495/Cast/ReadVariableOp?
Jgcn_attention_30/core_gcn_61/batch_normalization_495/Cast_1/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_61_batch_normalization_495_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_61/batch_normalization_495/Cast_1/ReadVariableOp?
Jgcn_attention_30/core_gcn_61/batch_normalization_495/Cast_2/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_61_batch_normalization_495_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_61/batch_normalization_495/Cast_2/ReadVariableOp?
Jgcn_attention_30/core_gcn_61/batch_normalization_495/Cast_3/ReadVariableOpReadVariableOpSgcn_attention_30_core_gcn_61_batch_normalization_495_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_30/core_gcn_61/batch_normalization_495/Cast_3/ReadVariableOp?
Dgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2F
Dgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/add/y?
Bgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/addAddV2Rgcn_attention_30/core_gcn_61/batch_normalization_495/Cast_1/ReadVariableOp:value:0Mgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/add/y:output:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/add?
Dgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/RsqrtRsqrtFgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/add:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/Rsqrt?
Bgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/mulMulHgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/Rsqrt:y:0Rgcn_attention_30/core_gcn_61/batch_normalization_495/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/mul?
Dgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/mul_1Mul*gcn_attention_30/core_gcn_61/add_5/add:z:0Fgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2F
Dgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/mul_1?
Dgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/mul_2MulPgcn_attention_30/core_gcn_61/batch_normalization_495/Cast/ReadVariableOp:value:0Fgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/mul:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/mul_2?
Bgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/subSubRgcn_attention_30/core_gcn_61/batch_normalization_495/Cast_2/ReadVariableOp:value:0Hgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2D
Bgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/sub?
Dgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/add_1AddV2Hgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/mul_1:z:0Fgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2F
Dgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/add_1?
/gcn_attention_30/core_gcn_61/activation_12/ReluReluIgcn_attention_30/core_gcn_61/batch_normalization_494/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????21
/gcn_attention_30/core_gcn_61/activation_12/Relu?
/gcn_attention_30/core_gcn_61/activation_13/ReluReluHgcn_attention_30/core_gcn_61/batch_normalization_495/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????21
/gcn_attention_30/core_gcn_61/activation_13/Relu?
&gcn_attention_30/core_gcn_61/add_6/addAddV2*gcn_attention_30/core_gcn_60/add_2/add:z:0=gcn_attention_30/core_gcn_61/activation_13/Relu:activations:0*
T0*4
_output_shapes"
 :??????????????????2(
&gcn_attention_30/core_gcn_61/add_6/add?
&gcn_attention_30/core_gcn_61/add_7/addAddV2*gcn_attention_30/core_gcn_60/add_3/add:z:0=gcn_attention_30/core_gcn_61/activation_12/Relu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????2(
&gcn_attention_30/core_gcn_61/add_7/add?
$gcn_attention_30/activation_153/ReluRelu*gcn_attention_30/core_gcn_61/add_6/add:z:0*
T0*4
_output_shapes"
 :??????????????????2&
$gcn_attention_30/activation_153/Relu?
<gcn_attention_30/batch_normalization_483/Cast/ReadVariableOpReadVariableOpEgcn_attention_30_batch_normalization_483_cast_readvariableop_resource*
_output_shapes
:*
dtype02>
<gcn_attention_30/batch_normalization_483/Cast/ReadVariableOp?
>gcn_attention_30/batch_normalization_483/Cast_1/ReadVariableOpReadVariableOpGgcn_attention_30_batch_normalization_483_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_30/batch_normalization_483/Cast_1/ReadVariableOp?
>gcn_attention_30/batch_normalization_483/Cast_2/ReadVariableOpReadVariableOpGgcn_attention_30_batch_normalization_483_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_30/batch_normalization_483/Cast_2/ReadVariableOp?
>gcn_attention_30/batch_normalization_483/Cast_3/ReadVariableOpReadVariableOpGgcn_attention_30_batch_normalization_483_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_30/batch_normalization_483/Cast_3/ReadVariableOp?
8gcn_attention_30/batch_normalization_483/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2:
8gcn_attention_30/batch_normalization_483/batchnorm/add/y?
6gcn_attention_30/batch_normalization_483/batchnorm/addAddV2Fgcn_attention_30/batch_normalization_483/Cast_1/ReadVariableOp:value:0Agcn_attention_30/batch_normalization_483/batchnorm/add/y:output:0*
T0*
_output_shapes
:28
6gcn_attention_30/batch_normalization_483/batchnorm/add?
8gcn_attention_30/batch_normalization_483/batchnorm/RsqrtRsqrt:gcn_attention_30/batch_normalization_483/batchnorm/add:z:0*
T0*
_output_shapes
:2:
8gcn_attention_30/batch_normalization_483/batchnorm/Rsqrt?
6gcn_attention_30/batch_normalization_483/batchnorm/mulMul<gcn_attention_30/batch_normalization_483/batchnorm/Rsqrt:y:0Fgcn_attention_30/batch_normalization_483/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:28
6gcn_attention_30/batch_normalization_483/batchnorm/mul?
8gcn_attention_30/batch_normalization_483/batchnorm/mul_1Mul2gcn_attention_30/activation_153/Relu:activations:0:gcn_attention_30/batch_normalization_483/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2:
8gcn_attention_30/batch_normalization_483/batchnorm/mul_1?
8gcn_attention_30/batch_normalization_483/batchnorm/mul_2MulDgcn_attention_30/batch_normalization_483/Cast/ReadVariableOp:value:0:gcn_attention_30/batch_normalization_483/batchnorm/mul:z:0*
T0*
_output_shapes
:2:
8gcn_attention_30/batch_normalization_483/batchnorm/mul_2?
6gcn_attention_30/batch_normalization_483/batchnorm/subSubFgcn_attention_30/batch_normalization_483/Cast_2/ReadVariableOp:value:0<gcn_attention_30/batch_normalization_483/batchnorm/mul_2:z:0*
T0*
_output_shapes
:28
6gcn_attention_30/batch_normalization_483/batchnorm/sub?
8gcn_attention_30/batch_normalization_483/batchnorm/add_1AddV2<gcn_attention_30/batch_normalization_483/batchnorm/mul_1:z:0:gcn_attention_30/batch_normalization_483/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2:
8gcn_attention_30/batch_normalization_483/batchnorm/add_1?
%gcn_attention_30/dropout_363/IdentityIdentity<gcn_attention_30/batch_normalization_483/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????2'
%gcn_attention_30/dropout_363/Identity?
0gcn_attention_30/conv1d_247/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:22
0gcn_attention_30/conv1d_247/conv1d/dilation_rate?
/gcn_attention_30/conv1d_247/conv1d/filter_shapeConst*
_output_shapes
:*
dtype0*!
valueB"         21
/gcn_attention_30/conv1d_247/conv1d/filter_shape?
(gcn_attention_30/conv1d_247/conv1d/stackConst*
_output_shapes

:*
dtype0*!
valueB"      2*
(gcn_attention_30/conv1d_247/conv1d/stack?
(gcn_attention_30/conv1d_247/conv1d/ShapeShape.gcn_attention_30/dropout_363/Identity:output:0*
T0*
_output_shapes
:2*
(gcn_attention_30/conv1d_247/conv1d/Shape?
6gcn_attention_30/conv1d_247/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6gcn_attention_30/conv1d_247/conv1d/strided_slice/stack?
8gcn_attention_30/conv1d_247/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8gcn_attention_30/conv1d_247/conv1d/strided_slice/stack_1?
8gcn_attention_30/conv1d_247/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8gcn_attention_30/conv1d_247/conv1d/strided_slice/stack_2?
0gcn_attention_30/conv1d_247/conv1d/strided_sliceStridedSlice1gcn_attention_30/conv1d_247/conv1d/Shape:output:0?gcn_attention_30/conv1d_247/conv1d/strided_slice/stack:output:0Agcn_attention_30/conv1d_247/conv1d/strided_slice/stack_1:output:0Agcn_attention_30/conv1d_247/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0gcn_attention_30/conv1d_247/conv1d/strided_slice?
*gcn_attention_30/conv1d_247/conv1d/stack_1Pack9gcn_attention_30/conv1d_247/conv1d/strided_slice:output:0*
N*
T0*
_output_shapes
:2,
*gcn_attention_30/conv1d_247/conv1d/stack_1?
Wgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2Y
Wgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice/stack?
Ygcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2[
Ygcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice/stack_1?
Ygcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2[
Ygcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice/stack_2?
Qgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_sliceStridedSlice1gcn_attention_30/conv1d_247/conv1d/stack:output:0`gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice/stack:output:0bgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice/stack_1:output:0bgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2S
Qgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice?
Ygcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2[
Ygcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_1/stack?
[gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2]
[gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1?
[gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2]
[gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2?
Sgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_1StridedSlice1gcn_attention_30/conv1d_247/conv1d/stack:output:0bgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_1/stack:output:0dgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1:output:0dgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2U
Sgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_1?
Ggcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/addAddV23gcn_attention_30/conv1d_247/conv1d/stack_1:output:0Zgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice:output:0*
T0*
_output_shapes
:2I
Ggcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/add?
Igcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/add_1AddV2Kgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/add:z:0\gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_1:output:0*
T0*
_output_shapes
:2K
Igcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/add_1?
Ggcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/modFloorModMgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/add_1:z:09gcn_attention_30/conv1d_247/conv1d/dilation_rate:output:0*
T0*
_output_shapes
:2I
Ggcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/mod?
Ggcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/subSub9gcn_attention_30/conv1d_247/conv1d/dilation_rate:output:0Kgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/mod:z:0*
T0*
_output_shapes
:2I
Ggcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/sub?
Igcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/mod_1FloorModKgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/sub:z:09gcn_attention_30/conv1d_247/conv1d/dilation_rate:output:0*
T0*
_output_shapes
:2K
Igcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/mod_1?
Igcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/add_2AddV2\gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_1:output:0Mgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/mod_1:z:0*
T0*
_output_shapes
:2K
Igcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/add_2?
Ygcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2[
Ygcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_2/stack?
[gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2]
[gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1?
[gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2]
[gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2?
Sgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_2StridedSliceZgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice:output:0bgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_2/stack:output:0dgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1:output:0dgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2U
Sgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_2?
Ygcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2[
Ygcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_3/stack?
[gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2]
[gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1?
[gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2]
[gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2?
Sgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_3StridedSliceMgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/add_2:z:0bgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_3/stack:output:0dgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1:output:0dgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2U
Sgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_3?
Ngcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/paddings/0Pack\gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_2:output:0\gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2P
Ngcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/paddings/0?
Lgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/paddingsPackWgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/paddings/0:output:0*
N*
T0*
_output_shapes

:2N
Lgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/paddings?
Ygcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2[
Ygcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_4/stack?
[gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2]
[gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1?
[gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2]
[gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2?
Sgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_4StridedSliceMgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/mod_1:z:0bgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_4/stack:output:0dgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1:output:0dgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2U
Sgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_4?
Mgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/crops/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2O
Mgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/crops/0/0?
Kgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/crops/0PackVgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/crops/0/0:output:0\gcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/strided_slice_4:output:0*
N*
T0*
_output_shapes
:2M
Kgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/crops/0?
Igcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/cropsPackTgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/crops/0:output:0*
N*
T0*
_output_shapes

:2K
Igcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/crops?
8gcn_attention_30/conv1d_247/conv1d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8gcn_attention_30/conv1d_247/conv1d/strided_slice_1/stack?
:gcn_attention_30/conv1d_247/conv1d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:gcn_attention_30/conv1d_247/conv1d/strided_slice_1/stack_1?
:gcn_attention_30/conv1d_247/conv1d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:gcn_attention_30/conv1d_247/conv1d/strided_slice_1/stack_2?
2gcn_attention_30/conv1d_247/conv1d/strided_slice_1StridedSliceUgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/paddings:output:0Agcn_attention_30/conv1d_247/conv1d/strided_slice_1/stack:output:0Cgcn_attention_30/conv1d_247/conv1d/strided_slice_1/stack_1:output:0Cgcn_attention_30/conv1d_247/conv1d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:24
2gcn_attention_30/conv1d_247/conv1d/strided_slice_1?
4gcn_attention_30/conv1d_247/conv1d/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4gcn_attention_30/conv1d_247/conv1d/concat/concat_dim?
0gcn_attention_30/conv1d_247/conv1d/concat/concatIdentity;gcn_attention_30/conv1d_247/conv1d/strided_slice_1:output:0*
T0*
_output_shapes

:22
0gcn_attention_30/conv1d_247/conv1d/concat/concat?
8gcn_attention_30/conv1d_247/conv1d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8gcn_attention_30/conv1d_247/conv1d/strided_slice_2/stack?
:gcn_attention_30/conv1d_247/conv1d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:gcn_attention_30/conv1d_247/conv1d/strided_slice_2/stack_1?
:gcn_attention_30/conv1d_247/conv1d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:gcn_attention_30/conv1d_247/conv1d/strided_slice_2/stack_2?
2gcn_attention_30/conv1d_247/conv1d/strided_slice_2StridedSliceRgcn_attention_30/conv1d_247/conv1d/required_space_to_batch_paddings/crops:output:0Agcn_attention_30/conv1d_247/conv1d/strided_slice_2/stack:output:0Cgcn_attention_30/conv1d_247/conv1d/strided_slice_2/stack_1:output:0Cgcn_attention_30/conv1d_247/conv1d/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:24
2gcn_attention_30/conv1d_247/conv1d/strided_slice_2?
6gcn_attention_30/conv1d_247/conv1d/concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 28
6gcn_attention_30/conv1d_247/conv1d/concat_1/concat_dim?
2gcn_attention_30/conv1d_247/conv1d/concat_1/concatIdentity;gcn_attention_30/conv1d_247/conv1d/strided_slice_2:output:0*
T0*
_output_shapes

:24
2gcn_attention_30/conv1d_247/conv1d/concat_1/concat?
=gcn_attention_30/conv1d_247/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2?
=gcn_attention_30/conv1d_247/conv1d/SpaceToBatchND/block_shape?
1gcn_attention_30/conv1d_247/conv1d/SpaceToBatchNDSpaceToBatchND.gcn_attention_30/dropout_363/Identity:output:0Fgcn_attention_30/conv1d_247/conv1d/SpaceToBatchND/block_shape:output:09gcn_attention_30/conv1d_247/conv1d/concat/concat:output:0*
T0*4
_output_shapes"
 :??????????????????23
1gcn_attention_30/conv1d_247/conv1d/SpaceToBatchND?
1gcn_attention_30/conv1d_247/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????23
1gcn_attention_30/conv1d_247/conv1d/ExpandDims/dim?
-gcn_attention_30/conv1d_247/conv1d/ExpandDims
ExpandDims:gcn_attention_30/conv1d_247/conv1d/SpaceToBatchND:output:0:gcn_attention_30/conv1d_247/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2/
-gcn_attention_30/conv1d_247/conv1d/ExpandDims?
>gcn_attention_30/conv1d_247/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpGgcn_attention_30_conv1d_247_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02@
>gcn_attention_30/conv1d_247/conv1d/ExpandDims_1/ReadVariableOp?
3gcn_attention_30/conv1d_247/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3gcn_attention_30/conv1d_247/conv1d/ExpandDims_1/dim?
/gcn_attention_30/conv1d_247/conv1d/ExpandDims_1
ExpandDimsFgcn_attention_30/conv1d_247/conv1d/ExpandDims_1/ReadVariableOp:value:0<gcn_attention_30/conv1d_247/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:21
/gcn_attention_30/conv1d_247/conv1d/ExpandDims_1?
"gcn_attention_30/conv1d_247/conv1dConv2D6gcn_attention_30/conv1d_247/conv1d/ExpandDims:output:08gcn_attention_30/conv1d_247/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingVALID*
strides
2$
"gcn_attention_30/conv1d_247/conv1d?
*gcn_attention_30/conv1d_247/conv1d/SqueezeSqueeze+gcn_attention_30/conv1d_247/conv1d:output:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims

?????????2,
*gcn_attention_30/conv1d_247/conv1d/Squeeze?
=gcn_attention_30/conv1d_247/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2?
=gcn_attention_30/conv1d_247/conv1d/BatchToSpaceND/block_shape?
1gcn_attention_30/conv1d_247/conv1d/BatchToSpaceNDBatchToSpaceND3gcn_attention_30/conv1d_247/conv1d/Squeeze:output:0Fgcn_attention_30/conv1d_247/conv1d/BatchToSpaceND/block_shape:output:0;gcn_attention_30/conv1d_247/conv1d/concat_1/concat:output:0*
T0*4
_output_shapes"
 :??????????????????23
1gcn_attention_30/conv1d_247/conv1d/BatchToSpaceND?
2gcn_attention_30/conv1d_247/BiasAdd/ReadVariableOpReadVariableOp;gcn_attention_30_conv1d_247_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2gcn_attention_30/conv1d_247/BiasAdd/ReadVariableOp?
#gcn_attention_30/conv1d_247/BiasAddBiasAdd:gcn_attention_30/conv1d_247/conv1d/BatchToSpaceND:output:0:gcn_attention_30/conv1d_247/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2%
#gcn_attention_30/conv1d_247/BiasAddK
xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
x?
NotEqualNotEqualinput_text1
x:output:0*
T0*5
_output_shapes#
!:???????????????????*
incompatible_shape_error( 2

NotEqualy
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Any/reduction_indicesq
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*0
_output_shapes
:??????????????????2
Any?
IdentityIdentity,gcn_attention_30/conv1d_247/BiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?/
NoOpNoOp=^gcn_attention_30/batch_normalization_480/Cast/ReadVariableOp?^gcn_attention_30/batch_normalization_480/Cast_1/ReadVariableOp?^gcn_attention_30/batch_normalization_480/Cast_2/ReadVariableOp?^gcn_attention_30/batch_normalization_480/Cast_3/ReadVariableOpI^gcn_attention_30/batch_normalization_482/FusedBatchNormV3/ReadVariableOpK^gcn_attention_30/batch_normalization_482/FusedBatchNormV3/ReadVariableOp_18^gcn_attention_30/batch_normalization_482/ReadVariableOp:^gcn_attention_30/batch_normalization_482/ReadVariableOp_1=^gcn_attention_30/batch_normalization_483/Cast/ReadVariableOp?^gcn_attention_30/batch_normalization_483/Cast_1/ReadVariableOp?^gcn_attention_30/batch_normalization_483/Cast_2/ReadVariableOp?^gcn_attention_30/batch_normalization_483/Cast_3/ReadVariableOp3^gcn_attention_30/conv1d_240/BiasAdd/ReadVariableOp?^gcn_attention_30/conv1d_240/conv1d/ExpandDims_1/ReadVariableOp3^gcn_attention_30/conv1d_247/BiasAdd/ReadVariableOp?^gcn_attention_30/conv1d_247/conv1d/ExpandDims_1/ReadVariableOp3^gcn_attention_30/conv2d_121/BiasAdd/ReadVariableOp2^gcn_attention_30/conv2d_121/Conv2D/ReadVariableOpU^gcn_attention_30/core_gcn_60/batch_normalization_484/FusedBatchNormV3/ReadVariableOpW^gcn_attention_30/core_gcn_60/batch_normalization_484/FusedBatchNormV3/ReadVariableOp_1D^gcn_attention_30/core_gcn_60/batch_normalization_484/ReadVariableOpF^gcn_attention_30/core_gcn_60/batch_normalization_484/ReadVariableOp_1I^gcn_attention_30/core_gcn_60/batch_normalization_485/Cast/ReadVariableOpK^gcn_attention_30/core_gcn_60/batch_normalization_485/Cast_1/ReadVariableOpK^gcn_attention_30/core_gcn_60/batch_normalization_485/Cast_2/ReadVariableOpK^gcn_attention_30/core_gcn_60/batch_normalization_485/Cast_3/ReadVariableOpI^gcn_attention_30/core_gcn_60/batch_normalization_486/Cast/ReadVariableOpK^gcn_attention_30/core_gcn_60/batch_normalization_486/Cast_1/ReadVariableOpK^gcn_attention_30/core_gcn_60/batch_normalization_486/Cast_2/ReadVariableOpK^gcn_attention_30/core_gcn_60/batch_normalization_486/Cast_3/ReadVariableOpI^gcn_attention_30/core_gcn_60/batch_normalization_487/Cast/ReadVariableOpK^gcn_attention_30/core_gcn_60/batch_normalization_487/Cast_1/ReadVariableOpK^gcn_attention_30/core_gcn_60/batch_normalization_487/Cast_2/ReadVariableOpK^gcn_attention_30/core_gcn_60/batch_normalization_487/Cast_3/ReadVariableOpU^gcn_attention_30/core_gcn_60/batch_normalization_488/FusedBatchNormV3/ReadVariableOpW^gcn_attention_30/core_gcn_60/batch_normalization_488/FusedBatchNormV3/ReadVariableOp_1D^gcn_attention_30/core_gcn_60/batch_normalization_488/ReadVariableOpF^gcn_attention_30/core_gcn_60/batch_normalization_488/ReadVariableOp_1I^gcn_attention_30/core_gcn_60/batch_normalization_489/Cast/ReadVariableOpK^gcn_attention_30/core_gcn_60/batch_normalization_489/Cast_1/ReadVariableOpK^gcn_attention_30/core_gcn_60/batch_normalization_489/Cast_2/ReadVariableOpK^gcn_attention_30/core_gcn_60/batch_normalization_489/Cast_3/ReadVariableOp?^gcn_attention_30/core_gcn_60/conv1d_241/BiasAdd/ReadVariableOpK^gcn_attention_30/core_gcn_60/conv1d_241/conv1d/ExpandDims_1/ReadVariableOp?^gcn_attention_30/core_gcn_60/conv1d_242/BiasAdd/ReadVariableOpK^gcn_attention_30/core_gcn_60/conv1d_242/conv1d/ExpandDims_1/ReadVariableOp?^gcn_attention_30/core_gcn_60/conv1d_243/BiasAdd/ReadVariableOpK^gcn_attention_30/core_gcn_60/conv1d_243/conv1d/ExpandDims_1/ReadVariableOp?^gcn_attention_30/core_gcn_60/conv2d_122/BiasAdd/ReadVariableOp>^gcn_attention_30/core_gcn_60/conv2d_122/Conv2D/ReadVariableOpU^gcn_attention_30/core_gcn_61/batch_normalization_490/FusedBatchNormV3/ReadVariableOpW^gcn_attention_30/core_gcn_61/batch_normalization_490/FusedBatchNormV3/ReadVariableOp_1D^gcn_attention_30/core_gcn_61/batch_normalization_490/ReadVariableOpF^gcn_attention_30/core_gcn_61/batch_normalization_490/ReadVariableOp_1I^gcn_attention_30/core_gcn_61/batch_normalization_491/Cast/ReadVariableOpK^gcn_attention_30/core_gcn_61/batch_normalization_491/Cast_1/ReadVariableOpK^gcn_attention_30/core_gcn_61/batch_normalization_491/Cast_2/ReadVariableOpK^gcn_attention_30/core_gcn_61/batch_normalization_491/Cast_3/ReadVariableOpI^gcn_attention_30/core_gcn_61/batch_normalization_492/Cast/ReadVariableOpK^gcn_attention_30/core_gcn_61/batch_normalization_492/Cast_1/ReadVariableOpK^gcn_attention_30/core_gcn_61/batch_normalization_492/Cast_2/ReadVariableOpK^gcn_attention_30/core_gcn_61/batch_normalization_492/Cast_3/ReadVariableOpI^gcn_attention_30/core_gcn_61/batch_normalization_493/Cast/ReadVariableOpK^gcn_attention_30/core_gcn_61/batch_normalization_493/Cast_1/ReadVariableOpK^gcn_attention_30/core_gcn_61/batch_normalization_493/Cast_2/ReadVariableOpK^gcn_attention_30/core_gcn_61/batch_normalization_493/Cast_3/ReadVariableOpU^gcn_attention_30/core_gcn_61/batch_normalization_494/FusedBatchNormV3/ReadVariableOpW^gcn_attention_30/core_gcn_61/batch_normalization_494/FusedBatchNormV3/ReadVariableOp_1D^gcn_attention_30/core_gcn_61/batch_normalization_494/ReadVariableOpF^gcn_attention_30/core_gcn_61/batch_normalization_494/ReadVariableOp_1I^gcn_attention_30/core_gcn_61/batch_normalization_495/Cast/ReadVariableOpK^gcn_attention_30/core_gcn_61/batch_normalization_495/Cast_1/ReadVariableOpK^gcn_attention_30/core_gcn_61/batch_normalization_495/Cast_2/ReadVariableOpK^gcn_attention_30/core_gcn_61/batch_normalization_495/Cast_3/ReadVariableOp?^gcn_attention_30/core_gcn_61/conv1d_244/BiasAdd/ReadVariableOpK^gcn_attention_30/core_gcn_61/conv1d_244/conv1d/ExpandDims_1/ReadVariableOp?^gcn_attention_30/core_gcn_61/conv1d_245/BiasAdd/ReadVariableOpK^gcn_attention_30/core_gcn_61/conv1d_245/conv1d/ExpandDims_1/ReadVariableOp?^gcn_attention_30/core_gcn_61/conv1d_246/BiasAdd/ReadVariableOpK^gcn_attention_30/core_gcn_61/conv1d_246/conv1d/ExpandDims_1/ReadVariableOp?^gcn_attention_30/core_gcn_61/conv2d_123/BiasAdd/ReadVariableOp>^gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/ReadVariableOp3^gcn_attention_30/dense_30/Tensordot/ReadVariableOp/^gcn_attention_30/embedding_30/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????????????:+???????????????????????????:+???????????????????????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2|
<gcn_attention_30/batch_normalization_480/Cast/ReadVariableOp<gcn_attention_30/batch_normalization_480/Cast/ReadVariableOp2?
>gcn_attention_30/batch_normalization_480/Cast_1/ReadVariableOp>gcn_attention_30/batch_normalization_480/Cast_1/ReadVariableOp2?
>gcn_attention_30/batch_normalization_480/Cast_2/ReadVariableOp>gcn_attention_30/batch_normalization_480/Cast_2/ReadVariableOp2?
>gcn_attention_30/batch_normalization_480/Cast_3/ReadVariableOp>gcn_attention_30/batch_normalization_480/Cast_3/ReadVariableOp2?
Hgcn_attention_30/batch_normalization_482/FusedBatchNormV3/ReadVariableOpHgcn_attention_30/batch_normalization_482/FusedBatchNormV3/ReadVariableOp2?
Jgcn_attention_30/batch_normalization_482/FusedBatchNormV3/ReadVariableOp_1Jgcn_attention_30/batch_normalization_482/FusedBatchNormV3/ReadVariableOp_12r
7gcn_attention_30/batch_normalization_482/ReadVariableOp7gcn_attention_30/batch_normalization_482/ReadVariableOp2v
9gcn_attention_30/batch_normalization_482/ReadVariableOp_19gcn_attention_30/batch_normalization_482/ReadVariableOp_12|
<gcn_attention_30/batch_normalization_483/Cast/ReadVariableOp<gcn_attention_30/batch_normalization_483/Cast/ReadVariableOp2?
>gcn_attention_30/batch_normalization_483/Cast_1/ReadVariableOp>gcn_attention_30/batch_normalization_483/Cast_1/ReadVariableOp2?
>gcn_attention_30/batch_normalization_483/Cast_2/ReadVariableOp>gcn_attention_30/batch_normalization_483/Cast_2/ReadVariableOp2?
>gcn_attention_30/batch_normalization_483/Cast_3/ReadVariableOp>gcn_attention_30/batch_normalization_483/Cast_3/ReadVariableOp2h
2gcn_attention_30/conv1d_240/BiasAdd/ReadVariableOp2gcn_attention_30/conv1d_240/BiasAdd/ReadVariableOp2?
>gcn_attention_30/conv1d_240/conv1d/ExpandDims_1/ReadVariableOp>gcn_attention_30/conv1d_240/conv1d/ExpandDims_1/ReadVariableOp2h
2gcn_attention_30/conv1d_247/BiasAdd/ReadVariableOp2gcn_attention_30/conv1d_247/BiasAdd/ReadVariableOp2?
>gcn_attention_30/conv1d_247/conv1d/ExpandDims_1/ReadVariableOp>gcn_attention_30/conv1d_247/conv1d/ExpandDims_1/ReadVariableOp2h
2gcn_attention_30/conv2d_121/BiasAdd/ReadVariableOp2gcn_attention_30/conv2d_121/BiasAdd/ReadVariableOp2f
1gcn_attention_30/conv2d_121/Conv2D/ReadVariableOp1gcn_attention_30/conv2d_121/Conv2D/ReadVariableOp2?
Tgcn_attention_30/core_gcn_60/batch_normalization_484/FusedBatchNormV3/ReadVariableOpTgcn_attention_30/core_gcn_60/batch_normalization_484/FusedBatchNormV3/ReadVariableOp2?
Vgcn_attention_30/core_gcn_60/batch_normalization_484/FusedBatchNormV3/ReadVariableOp_1Vgcn_attention_30/core_gcn_60/batch_normalization_484/FusedBatchNormV3/ReadVariableOp_12?
Cgcn_attention_30/core_gcn_60/batch_normalization_484/ReadVariableOpCgcn_attention_30/core_gcn_60/batch_normalization_484/ReadVariableOp2?
Egcn_attention_30/core_gcn_60/batch_normalization_484/ReadVariableOp_1Egcn_attention_30/core_gcn_60/batch_normalization_484/ReadVariableOp_12?
Hgcn_attention_30/core_gcn_60/batch_normalization_485/Cast/ReadVariableOpHgcn_attention_30/core_gcn_60/batch_normalization_485/Cast/ReadVariableOp2?
Jgcn_attention_30/core_gcn_60/batch_normalization_485/Cast_1/ReadVariableOpJgcn_attention_30/core_gcn_60/batch_normalization_485/Cast_1/ReadVariableOp2?
Jgcn_attention_30/core_gcn_60/batch_normalization_485/Cast_2/ReadVariableOpJgcn_attention_30/core_gcn_60/batch_normalization_485/Cast_2/ReadVariableOp2?
Jgcn_attention_30/core_gcn_60/batch_normalization_485/Cast_3/ReadVariableOpJgcn_attention_30/core_gcn_60/batch_normalization_485/Cast_3/ReadVariableOp2?
Hgcn_attention_30/core_gcn_60/batch_normalization_486/Cast/ReadVariableOpHgcn_attention_30/core_gcn_60/batch_normalization_486/Cast/ReadVariableOp2?
Jgcn_attention_30/core_gcn_60/batch_normalization_486/Cast_1/ReadVariableOpJgcn_attention_30/core_gcn_60/batch_normalization_486/Cast_1/ReadVariableOp2?
Jgcn_attention_30/core_gcn_60/batch_normalization_486/Cast_2/ReadVariableOpJgcn_attention_30/core_gcn_60/batch_normalization_486/Cast_2/ReadVariableOp2?
Jgcn_attention_30/core_gcn_60/batch_normalization_486/Cast_3/ReadVariableOpJgcn_attention_30/core_gcn_60/batch_normalization_486/Cast_3/ReadVariableOp2?
Hgcn_attention_30/core_gcn_60/batch_normalization_487/Cast/ReadVariableOpHgcn_attention_30/core_gcn_60/batch_normalization_487/Cast/ReadVariableOp2?
Jgcn_attention_30/core_gcn_60/batch_normalization_487/Cast_1/ReadVariableOpJgcn_attention_30/core_gcn_60/batch_normalization_487/Cast_1/ReadVariableOp2?
Jgcn_attention_30/core_gcn_60/batch_normalization_487/Cast_2/ReadVariableOpJgcn_attention_30/core_gcn_60/batch_normalization_487/Cast_2/ReadVariableOp2?
Jgcn_attention_30/core_gcn_60/batch_normalization_487/Cast_3/ReadVariableOpJgcn_attention_30/core_gcn_60/batch_normalization_487/Cast_3/ReadVariableOp2?
Tgcn_attention_30/core_gcn_60/batch_normalization_488/FusedBatchNormV3/ReadVariableOpTgcn_attention_30/core_gcn_60/batch_normalization_488/FusedBatchNormV3/ReadVariableOp2?
Vgcn_attention_30/core_gcn_60/batch_normalization_488/FusedBatchNormV3/ReadVariableOp_1Vgcn_attention_30/core_gcn_60/batch_normalization_488/FusedBatchNormV3/ReadVariableOp_12?
Cgcn_attention_30/core_gcn_60/batch_normalization_488/ReadVariableOpCgcn_attention_30/core_gcn_60/batch_normalization_488/ReadVariableOp2?
Egcn_attention_30/core_gcn_60/batch_normalization_488/ReadVariableOp_1Egcn_attention_30/core_gcn_60/batch_normalization_488/ReadVariableOp_12?
Hgcn_attention_30/core_gcn_60/batch_normalization_489/Cast/ReadVariableOpHgcn_attention_30/core_gcn_60/batch_normalization_489/Cast/ReadVariableOp2?
Jgcn_attention_30/core_gcn_60/batch_normalization_489/Cast_1/ReadVariableOpJgcn_attention_30/core_gcn_60/batch_normalization_489/Cast_1/ReadVariableOp2?
Jgcn_attention_30/core_gcn_60/batch_normalization_489/Cast_2/ReadVariableOpJgcn_attention_30/core_gcn_60/batch_normalization_489/Cast_2/ReadVariableOp2?
Jgcn_attention_30/core_gcn_60/batch_normalization_489/Cast_3/ReadVariableOpJgcn_attention_30/core_gcn_60/batch_normalization_489/Cast_3/ReadVariableOp2?
>gcn_attention_30/core_gcn_60/conv1d_241/BiasAdd/ReadVariableOp>gcn_attention_30/core_gcn_60/conv1d_241/BiasAdd/ReadVariableOp2?
Jgcn_attention_30/core_gcn_60/conv1d_241/conv1d/ExpandDims_1/ReadVariableOpJgcn_attention_30/core_gcn_60/conv1d_241/conv1d/ExpandDims_1/ReadVariableOp2?
>gcn_attention_30/core_gcn_60/conv1d_242/BiasAdd/ReadVariableOp>gcn_attention_30/core_gcn_60/conv1d_242/BiasAdd/ReadVariableOp2?
Jgcn_attention_30/core_gcn_60/conv1d_242/conv1d/ExpandDims_1/ReadVariableOpJgcn_attention_30/core_gcn_60/conv1d_242/conv1d/ExpandDims_1/ReadVariableOp2?
>gcn_attention_30/core_gcn_60/conv1d_243/BiasAdd/ReadVariableOp>gcn_attention_30/core_gcn_60/conv1d_243/BiasAdd/ReadVariableOp2?
Jgcn_attention_30/core_gcn_60/conv1d_243/conv1d/ExpandDims_1/ReadVariableOpJgcn_attention_30/core_gcn_60/conv1d_243/conv1d/ExpandDims_1/ReadVariableOp2?
>gcn_attention_30/core_gcn_60/conv2d_122/BiasAdd/ReadVariableOp>gcn_attention_30/core_gcn_60/conv2d_122/BiasAdd/ReadVariableOp2~
=gcn_attention_30/core_gcn_60/conv2d_122/Conv2D/ReadVariableOp=gcn_attention_30/core_gcn_60/conv2d_122/Conv2D/ReadVariableOp2?
Tgcn_attention_30/core_gcn_61/batch_normalization_490/FusedBatchNormV3/ReadVariableOpTgcn_attention_30/core_gcn_61/batch_normalization_490/FusedBatchNormV3/ReadVariableOp2?
Vgcn_attention_30/core_gcn_61/batch_normalization_490/FusedBatchNormV3/ReadVariableOp_1Vgcn_attention_30/core_gcn_61/batch_normalization_490/FusedBatchNormV3/ReadVariableOp_12?
Cgcn_attention_30/core_gcn_61/batch_normalization_490/ReadVariableOpCgcn_attention_30/core_gcn_61/batch_normalization_490/ReadVariableOp2?
Egcn_attention_30/core_gcn_61/batch_normalization_490/ReadVariableOp_1Egcn_attention_30/core_gcn_61/batch_normalization_490/ReadVariableOp_12?
Hgcn_attention_30/core_gcn_61/batch_normalization_491/Cast/ReadVariableOpHgcn_attention_30/core_gcn_61/batch_normalization_491/Cast/ReadVariableOp2?
Jgcn_attention_30/core_gcn_61/batch_normalization_491/Cast_1/ReadVariableOpJgcn_attention_30/core_gcn_61/batch_normalization_491/Cast_1/ReadVariableOp2?
Jgcn_attention_30/core_gcn_61/batch_normalization_491/Cast_2/ReadVariableOpJgcn_attention_30/core_gcn_61/batch_normalization_491/Cast_2/ReadVariableOp2?
Jgcn_attention_30/core_gcn_61/batch_normalization_491/Cast_3/ReadVariableOpJgcn_attention_30/core_gcn_61/batch_normalization_491/Cast_3/ReadVariableOp2?
Hgcn_attention_30/core_gcn_61/batch_normalization_492/Cast/ReadVariableOpHgcn_attention_30/core_gcn_61/batch_normalization_492/Cast/ReadVariableOp2?
Jgcn_attention_30/core_gcn_61/batch_normalization_492/Cast_1/ReadVariableOpJgcn_attention_30/core_gcn_61/batch_normalization_492/Cast_1/ReadVariableOp2?
Jgcn_attention_30/core_gcn_61/batch_normalization_492/Cast_2/ReadVariableOpJgcn_attention_30/core_gcn_61/batch_normalization_492/Cast_2/ReadVariableOp2?
Jgcn_attention_30/core_gcn_61/batch_normalization_492/Cast_3/ReadVariableOpJgcn_attention_30/core_gcn_61/batch_normalization_492/Cast_3/ReadVariableOp2?
Hgcn_attention_30/core_gcn_61/batch_normalization_493/Cast/ReadVariableOpHgcn_attention_30/core_gcn_61/batch_normalization_493/Cast/ReadVariableOp2?
Jgcn_attention_30/core_gcn_61/batch_normalization_493/Cast_1/ReadVariableOpJgcn_attention_30/core_gcn_61/batch_normalization_493/Cast_1/ReadVariableOp2?
Jgcn_attention_30/core_gcn_61/batch_normalization_493/Cast_2/ReadVariableOpJgcn_attention_30/core_gcn_61/batch_normalization_493/Cast_2/ReadVariableOp2?
Jgcn_attention_30/core_gcn_61/batch_normalization_493/Cast_3/ReadVariableOpJgcn_attention_30/core_gcn_61/batch_normalization_493/Cast_3/ReadVariableOp2?
Tgcn_attention_30/core_gcn_61/batch_normalization_494/FusedBatchNormV3/ReadVariableOpTgcn_attention_30/core_gcn_61/batch_normalization_494/FusedBatchNormV3/ReadVariableOp2?
Vgcn_attention_30/core_gcn_61/batch_normalization_494/FusedBatchNormV3/ReadVariableOp_1Vgcn_attention_30/core_gcn_61/batch_normalization_494/FusedBatchNormV3/ReadVariableOp_12?
Cgcn_attention_30/core_gcn_61/batch_normalization_494/ReadVariableOpCgcn_attention_30/core_gcn_61/batch_normalization_494/ReadVariableOp2?
Egcn_attention_30/core_gcn_61/batch_normalization_494/ReadVariableOp_1Egcn_attention_30/core_gcn_61/batch_normalization_494/ReadVariableOp_12?
Hgcn_attention_30/core_gcn_61/batch_normalization_495/Cast/ReadVariableOpHgcn_attention_30/core_gcn_61/batch_normalization_495/Cast/ReadVariableOp2?
Jgcn_attention_30/core_gcn_61/batch_normalization_495/Cast_1/ReadVariableOpJgcn_attention_30/core_gcn_61/batch_normalization_495/Cast_1/ReadVariableOp2?
Jgcn_attention_30/core_gcn_61/batch_normalization_495/Cast_2/ReadVariableOpJgcn_attention_30/core_gcn_61/batch_normalization_495/Cast_2/ReadVariableOp2?
Jgcn_attention_30/core_gcn_61/batch_normalization_495/Cast_3/ReadVariableOpJgcn_attention_30/core_gcn_61/batch_normalization_495/Cast_3/ReadVariableOp2?
>gcn_attention_30/core_gcn_61/conv1d_244/BiasAdd/ReadVariableOp>gcn_attention_30/core_gcn_61/conv1d_244/BiasAdd/ReadVariableOp2?
Jgcn_attention_30/core_gcn_61/conv1d_244/conv1d/ExpandDims_1/ReadVariableOpJgcn_attention_30/core_gcn_61/conv1d_244/conv1d/ExpandDims_1/ReadVariableOp2?
>gcn_attention_30/core_gcn_61/conv1d_245/BiasAdd/ReadVariableOp>gcn_attention_30/core_gcn_61/conv1d_245/BiasAdd/ReadVariableOp2?
Jgcn_attention_30/core_gcn_61/conv1d_245/conv1d/ExpandDims_1/ReadVariableOpJgcn_attention_30/core_gcn_61/conv1d_245/conv1d/ExpandDims_1/ReadVariableOp2?
>gcn_attention_30/core_gcn_61/conv1d_246/BiasAdd/ReadVariableOp>gcn_attention_30/core_gcn_61/conv1d_246/BiasAdd/ReadVariableOp2?
Jgcn_attention_30/core_gcn_61/conv1d_246/conv1d/ExpandDims_1/ReadVariableOpJgcn_attention_30/core_gcn_61/conv1d_246/conv1d/ExpandDims_1/ReadVariableOp2?
>gcn_attention_30/core_gcn_61/conv2d_123/BiasAdd/ReadVariableOp>gcn_attention_30/core_gcn_61/conv2d_123/BiasAdd/ReadVariableOp2~
=gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/ReadVariableOp=gcn_attention_30/core_gcn_61/conv2d_123/Conv2D/ReadVariableOp2h
2gcn_attention_30/dense_30/Tensordot/ReadVariableOp2gcn_attention_30/dense_30/Tensordot/ReadVariableOp2`
.gcn_attention_30/embedding_30/embedding_lookup.gcn_attention_30/embedding_30/embedding_lookup:b ^
5
_output_shapes#
!:???????????????????
%
_user_specified_nameinput_text1:nj
A
_output_shapes/
-:+???????????????????????????
%
_user_specified_nameinput_text2:nj
A
_output_shapes/
-:+???????????????????????????
%
_user_specified_nameinput_text3
?
?
S__inference_batch_normalization_492_layer_call_and_return_conditional_losses_378786

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_482_layer_call_fn_379503

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????	*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_482_layer_call_and_return_conditional_losses_3773082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????	
 
_user_specified_nameinputs
?*
?
S__inference_batch_normalization_495_layer_call_and_return_conditional_losses_379296

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_483_layer_call_and_return_conditional_losses_379598

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_491_layer_call_fn_380208

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_491_layer_call_and_return_conditional_losses_3786242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_493_layer_call_and_return_conditional_losses_380401

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?*
?
S__inference_batch_normalization_493_layer_call_and_return_conditional_losses_380435

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_490_layer_call_and_return_conditional_losses_378496

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_379654g
Mgcn_attention_30_conv2d_121_kernel_regularizer_square_readvariableop_resource:		
identity??Dgcn_attention_30/conv2d_121/kernel/Regularizer/Square/ReadVariableOp?
Dgcn_attention_30/conv2d_121/kernel/Regularizer/Square/ReadVariableOpReadVariableOpMgcn_attention_30_conv2d_121_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:		*
dtype02F
Dgcn_attention_30/conv2d_121/kernel/Regularizer/Square/ReadVariableOp?
5gcn_attention_30/conv2d_121/kernel/Regularizer/SquareSquareLgcn_attention_30/conv2d_121/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		27
5gcn_attention_30/conv2d_121/kernel/Regularizer/Square?
4gcn_attention_30/conv2d_121/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             26
4gcn_attention_30/conv2d_121/kernel/Regularizer/Const?
2gcn_attention_30/conv2d_121/kernel/Regularizer/SumSum9gcn_attention_30/conv2d_121/kernel/Regularizer/Square:y:0=gcn_attention_30/conv2d_121/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 24
2gcn_attention_30/conv2d_121/kernel/Regularizer/Sum?
4gcn_attention_30/conv2d_121/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??826
4gcn_attention_30/conv2d_121/kernel/Regularizer/mul/x?
2gcn_attention_30/conv2d_121/kernel/Regularizer/mulMul=gcn_attention_30/conv2d_121/kernel/Regularizer/mul/x:output:0;gcn_attention_30/conv2d_121/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 24
2gcn_attention_30/conv2d_121/kernel/Regularizer/mul?
IdentityIdentity6gcn_attention_30/conv2d_121/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOpE^gcn_attention_30/conv2d_121/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2?
Dgcn_attention_30/conv2d_121/kernel/Regularizer/Square/ReadVariableOpDgcn_attention_30/conv2d_121/kernel/Regularizer/Square/ReadVariableOp
?
?
S__inference_batch_normalization_495_layer_call_and_return_conditional_losses_380543

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?*
?
S__inference_batch_normalization_489_layer_call_and_return_conditional_losses_378396

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?*
?
S__inference_batch_normalization_487_layer_call_and_return_conditional_losses_378108

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_490_layer_call_and_return_conditional_losses_378540

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_484_layer_call_and_return_conditional_losses_377596

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_482_layer_call_and_return_conditional_losses_379552

inputs%
readvariableop_resource:	'
readvariableop_1_resource:	6
(fusedbatchnormv3_readvariableop_resource:	8
*fusedbatchnormv3_readvariableop_1_resource:	
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:	*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:	*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????	:	:	:	:	:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????	2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????	: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????	
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_487_layer_call_fn_379920

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_487_layer_call_and_return_conditional_losses_3781082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?*
?
S__inference_batch_normalization_491_layer_call_and_return_conditional_losses_380275

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_480_layer_call_and_return_conditional_losses_377148

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?*
?
S__inference_batch_normalization_485_layer_call_and_return_conditional_losses_377784

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_482_layer_call_fn_379516

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_482_layer_call_and_return_conditional_losses_3773522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????	
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_492_layer_call_and_return_conditional_losses_380321

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_480_layer_call_fn_379417

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_480_layer_call_and_return_conditional_losses_3771482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?*
?
S__inference_batch_normalization_480_layer_call_and_return_conditional_losses_379484

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?*
?
S__inference_batch_normalization_487_layer_call_and_return_conditional_losses_379974

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?*
?
S__inference_batch_normalization_485_layer_call_and_return_conditional_losses_379814

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_494_layer_call_and_return_conditional_losses_379152

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_482_layer_call_and_return_conditional_losses_377308

inputs%
readvariableop_resource:	'
readvariableop_1_resource:	6
(fusedbatchnormv3_readvariableop_resource:	8
*fusedbatchnormv3_readvariableop_1_resource:	
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:	*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:	*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????	:	:	:	:	:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????	2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????	: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????	
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_494_layer_call_and_return_conditional_losses_380497

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_488_layer_call_fn_380000

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_488_layer_call_and_return_conditional_losses_3782522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_486_layer_call_and_return_conditional_losses_379860

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?*
?
S__inference_batch_normalization_495_layer_call_and_return_conditional_losses_380577

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_493_layer_call_fn_380381

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_493_layer_call_and_return_conditional_losses_3790082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_485_layer_call_fn_379747

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_485_layer_call_and_return_conditional_losses_3777242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_495_layer_call_and_return_conditional_losses_379236

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_485_layer_call_and_return_conditional_losses_377724

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_490_layer_call_and_return_conditional_losses_380177

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_485_layer_call_and_return_conditional_losses_379780

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_494_layer_call_and_return_conditional_losses_379108

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_480_layer_call_fn_379430

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_480_layer_call_and_return_conditional_losses_3772082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
??
?J
"__inference__traced_restore_381127
file_prefixJ
3assignvariableop_gcn_attention_30_conv1d_240_kernel:?A
3assignvariableop_1_gcn_attention_30_conv1d_240_bias:O
Aassignvariableop_2_gcn_attention_30_batch_normalization_480_gamma:N
@assignvariableop_3_gcn_attention_30_batch_normalization_480_beta:U
Gassignvariableop_4_gcn_attention_30_batch_normalization_480_moving_mean:Y
Kassignvariableop_5_gcn_attention_30_batch_normalization_480_moving_variance:E
3assignvariableop_6_gcn_attention_30_dense_30_kernel:	O
5assignvariableop_7_gcn_attention_30_conv2d_121_kernel:		A
3assignvariableop_8_gcn_attention_30_conv2d_121_bias:	O
Aassignvariableop_9_gcn_attention_30_batch_normalization_482_gamma:	O
Aassignvariableop_10_gcn_attention_30_batch_normalization_482_beta:	V
Hassignvariableop_11_gcn_attention_30_batch_normalization_482_moving_mean:	Z
Lassignvariableop_12_gcn_attention_30_batch_normalization_482_moving_variance:	N
<assignvariableop_13_gcn_attention_30_embedding_30_embeddings:	P
Bassignvariableop_14_gcn_attention_30_batch_normalization_483_gamma:O
Aassignvariableop_15_gcn_attention_30_batch_normalization_483_beta:V
Hassignvariableop_16_gcn_attention_30_batch_normalization_483_moving_mean:Z
Lassignvariableop_17_gcn_attention_30_batch_normalization_483_moving_variance:L
6assignvariableop_18_gcn_attention_30_conv1d_247_kernel:B
4assignvariableop_19_gcn_attention_30_conv1d_247_bias:\
Bassignvariableop_20_gcn_attention_30_core_gcn_60_conv2d_122_kernel:N
@assignvariableop_21_gcn_attention_30_core_gcn_60_conv2d_122_bias:\
Nassignvariableop_22_gcn_attention_30_core_gcn_60_batch_normalization_484_gamma:[
Massignvariableop_23_gcn_attention_30_core_gcn_60_batch_normalization_484_beta:X
Bassignvariableop_24_gcn_attention_30_core_gcn_60_conv1d_241_kernel:N
@assignvariableop_25_gcn_attention_30_core_gcn_60_conv1d_241_bias:\
Nassignvariableop_26_gcn_attention_30_core_gcn_60_batch_normalization_485_gamma:[
Massignvariableop_27_gcn_attention_30_core_gcn_60_batch_normalization_485_beta:X
Bassignvariableop_28_gcn_attention_30_core_gcn_60_conv1d_242_kernel:N
@assignvariableop_29_gcn_attention_30_core_gcn_60_conv1d_242_bias:\
Nassignvariableop_30_gcn_attention_30_core_gcn_60_batch_normalization_486_gamma:[
Massignvariableop_31_gcn_attention_30_core_gcn_60_batch_normalization_486_beta:X
Bassignvariableop_32_gcn_attention_30_core_gcn_60_conv1d_243_kernel:N
@assignvariableop_33_gcn_attention_30_core_gcn_60_conv1d_243_bias:\
Nassignvariableop_34_gcn_attention_30_core_gcn_60_batch_normalization_487_gamma:[
Massignvariableop_35_gcn_attention_30_core_gcn_60_batch_normalization_487_beta:\
Nassignvariableop_36_gcn_attention_30_core_gcn_60_batch_normalization_488_gamma:[
Massignvariableop_37_gcn_attention_30_core_gcn_60_batch_normalization_488_beta:\
Nassignvariableop_38_gcn_attention_30_core_gcn_60_batch_normalization_489_gamma:[
Massignvariableop_39_gcn_attention_30_core_gcn_60_batch_normalization_489_beta:\
Bassignvariableop_40_gcn_attention_30_core_gcn_61_conv2d_123_kernel:N
@assignvariableop_41_gcn_attention_30_core_gcn_61_conv2d_123_bias:\
Nassignvariableop_42_gcn_attention_30_core_gcn_61_batch_normalization_490_gamma:[
Massignvariableop_43_gcn_attention_30_core_gcn_61_batch_normalization_490_beta:X
Bassignvariableop_44_gcn_attention_30_core_gcn_61_conv1d_244_kernel:N
@assignvariableop_45_gcn_attention_30_core_gcn_61_conv1d_244_bias:\
Nassignvariableop_46_gcn_attention_30_core_gcn_61_batch_normalization_491_gamma:[
Massignvariableop_47_gcn_attention_30_core_gcn_61_batch_normalization_491_beta:X
Bassignvariableop_48_gcn_attention_30_core_gcn_61_conv1d_245_kernel:N
@assignvariableop_49_gcn_attention_30_core_gcn_61_conv1d_245_bias:\
Nassignvariableop_50_gcn_attention_30_core_gcn_61_batch_normalization_492_gamma:[
Massignvariableop_51_gcn_attention_30_core_gcn_61_batch_normalization_492_beta:X
Bassignvariableop_52_gcn_attention_30_core_gcn_61_conv1d_246_kernel:N
@assignvariableop_53_gcn_attention_30_core_gcn_61_conv1d_246_bias:\
Nassignvariableop_54_gcn_attention_30_core_gcn_61_batch_normalization_493_gamma:[
Massignvariableop_55_gcn_attention_30_core_gcn_61_batch_normalization_493_beta:\
Nassignvariableop_56_gcn_attention_30_core_gcn_61_batch_normalization_494_gamma:[
Massignvariableop_57_gcn_attention_30_core_gcn_61_batch_normalization_494_beta:\
Nassignvariableop_58_gcn_attention_30_core_gcn_61_batch_normalization_495_gamma:[
Massignvariableop_59_gcn_attention_30_core_gcn_61_batch_normalization_495_beta:b
Tassignvariableop_60_gcn_attention_30_core_gcn_60_batch_normalization_484_moving_mean:f
Xassignvariableop_61_gcn_attention_30_core_gcn_60_batch_normalization_484_moving_variance:b
Tassignvariableop_62_gcn_attention_30_core_gcn_60_batch_normalization_485_moving_mean:f
Xassignvariableop_63_gcn_attention_30_core_gcn_60_batch_normalization_485_moving_variance:b
Tassignvariableop_64_gcn_attention_30_core_gcn_60_batch_normalization_486_moving_mean:f
Xassignvariableop_65_gcn_attention_30_core_gcn_60_batch_normalization_486_moving_variance:b
Tassignvariableop_66_gcn_attention_30_core_gcn_60_batch_normalization_487_moving_mean:f
Xassignvariableop_67_gcn_attention_30_core_gcn_60_batch_normalization_487_moving_variance:b
Tassignvariableop_68_gcn_attention_30_core_gcn_60_batch_normalization_488_moving_mean:f
Xassignvariableop_69_gcn_attention_30_core_gcn_60_batch_normalization_488_moving_variance:b
Tassignvariableop_70_gcn_attention_30_core_gcn_60_batch_normalization_489_moving_mean:f
Xassignvariableop_71_gcn_attention_30_core_gcn_60_batch_normalization_489_moving_variance:b
Tassignvariableop_72_gcn_attention_30_core_gcn_61_batch_normalization_490_moving_mean:f
Xassignvariableop_73_gcn_attention_30_core_gcn_61_batch_normalization_490_moving_variance:b
Tassignvariableop_74_gcn_attention_30_core_gcn_61_batch_normalization_491_moving_mean:f
Xassignvariableop_75_gcn_attention_30_core_gcn_61_batch_normalization_491_moving_variance:b
Tassignvariableop_76_gcn_attention_30_core_gcn_61_batch_normalization_492_moving_mean:f
Xassignvariableop_77_gcn_attention_30_core_gcn_61_batch_normalization_492_moving_variance:b
Tassignvariableop_78_gcn_attention_30_core_gcn_61_batch_normalization_493_moving_mean:f
Xassignvariableop_79_gcn_attention_30_core_gcn_61_batch_normalization_493_moving_variance:b
Tassignvariableop_80_gcn_attention_30_core_gcn_61_batch_normalization_494_moving_mean:f
Xassignvariableop_81_gcn_attention_30_core_gcn_61_batch_normalization_494_moving_variance:b
Tassignvariableop_82_gcn_attention_30_core_gcn_61_batch_normalization_495_moving_mean:f
Xassignvariableop_83_gcn_attention_30_core_gcn_61_batch_normalization_495_moving_variance:
identity_85??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_9?,
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:U*
dtype0*?+
value?+B?+UBCself_attention_layer/node0_Conv1D/kernel/.ATTRIBUTES/VARIABLE_VALUEBAself_attention_layer/node0_Conv1D/bias/.ATTRIBUTES/VARIABLE_VALUEBNself_attention_layer/node0_BatchNormalization/gamma/.ATTRIBUTES/VARIABLE_VALUEBMself_attention_layer/node0_BatchNormalization/beta/.ATTRIBUTES/VARIABLE_VALUEBTself_attention_layer/node0_BatchNormalization/moving_mean/.ATTRIBUTES/VARIABLE_VALUEBXself_attention_layer/node0_BatchNormalization/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBEself_attention_layer/distance_Dense/kernel/.ATTRIBUTES/VARIABLE_VALUEBMself_attention_layer/distance_Convolution2D/kernel/.ATTRIBUTES/VARIABLE_VALUEBKself_attention_layer/distance_Convolution2D/bias/.ATTRIBUTES/VARIABLE_VALUEBQself_attention_layer/distance_BatchNormalization/gamma/.ATTRIBUTES/VARIABLE_VALUEBPself_attention_layer/distance_BatchNormalization/beta/.ATTRIBUTES/VARIABLE_VALUEBWself_attention_layer/distance_BatchNormalization/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB[self_attention_layer/distance_BatchNormalization/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBDself_attention_layer/adj_Dense/embeddings/.ATTRIBUTES/VARIABLE_VALUEBNself_attention_layer/node_BatchNormalization5/gamma/.ATTRIBUTES/VARIABLE_VALUEBMself_attention_layer/node_BatchNormalization5/beta/.ATTRIBUTES/VARIABLE_VALUEBTself_attention_layer/node_BatchNormalization5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEBXself_attention_layer/node_BatchNormalization5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/node_Conv1D_out/kernel/.ATTRIBUTES/VARIABLE_VALUEBDself_attention_layer/node_Conv1D_out/bias/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/12/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/13/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/14/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/15/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/16/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/17/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/18/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/19/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/20/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/21/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/22/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/23/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/24/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/25/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/26/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/27/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/28/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/29/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/30/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/31/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/32/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/33/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/34/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/35/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/36/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/37/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/38/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/39/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/40/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/41/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/42/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/43/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/44/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/45/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/46/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/47/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/48/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/49/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/50/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/51/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/60/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/61/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/62/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/63/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/64/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/65/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/66/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/67/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/68/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/69/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/70/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/71/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/72/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/73/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/74/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/75/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/76/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/77/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/78/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/79/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/80/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/81/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/82/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/83/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:U*
dtype0*?
value?B?UB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*c
dtypesY
W2U2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp3assignvariableop_gcn_attention_30_conv1d_240_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp3assignvariableop_1_gcn_attention_30_conv1d_240_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpAassignvariableop_2_gcn_attention_30_batch_normalization_480_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp@assignvariableop_3_gcn_attention_30_batch_normalization_480_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpGassignvariableop_4_gcn_attention_30_batch_normalization_480_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpKassignvariableop_5_gcn_attention_30_batch_normalization_480_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp3assignvariableop_6_gcn_attention_30_dense_30_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp5assignvariableop_7_gcn_attention_30_conv2d_121_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp3assignvariableop_8_gcn_attention_30_conv2d_121_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpAassignvariableop_9_gcn_attention_30_batch_normalization_482_gammaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpAassignvariableop_10_gcn_attention_30_batch_normalization_482_betaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpHassignvariableop_11_gcn_attention_30_batch_normalization_482_moving_meanIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpLassignvariableop_12_gcn_attention_30_batch_normalization_482_moving_varianceIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp<assignvariableop_13_gcn_attention_30_embedding_30_embeddingsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpBassignvariableop_14_gcn_attention_30_batch_normalization_483_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpAassignvariableop_15_gcn_attention_30_batch_normalization_483_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpHassignvariableop_16_gcn_attention_30_batch_normalization_483_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpLassignvariableop_17_gcn_attention_30_batch_normalization_483_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp6assignvariableop_18_gcn_attention_30_conv1d_247_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp4assignvariableop_19_gcn_attention_30_conv1d_247_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpBassignvariableop_20_gcn_attention_30_core_gcn_60_conv2d_122_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp@assignvariableop_21_gcn_attention_30_core_gcn_60_conv2d_122_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpNassignvariableop_22_gcn_attention_30_core_gcn_60_batch_normalization_484_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpMassignvariableop_23_gcn_attention_30_core_gcn_60_batch_normalization_484_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpBassignvariableop_24_gcn_attention_30_core_gcn_60_conv1d_241_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp@assignvariableop_25_gcn_attention_30_core_gcn_60_conv1d_241_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpNassignvariableop_26_gcn_attention_30_core_gcn_60_batch_normalization_485_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpMassignvariableop_27_gcn_attention_30_core_gcn_60_batch_normalization_485_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpBassignvariableop_28_gcn_attention_30_core_gcn_60_conv1d_242_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp@assignvariableop_29_gcn_attention_30_core_gcn_60_conv1d_242_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpNassignvariableop_30_gcn_attention_30_core_gcn_60_batch_normalization_486_gammaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpMassignvariableop_31_gcn_attention_30_core_gcn_60_batch_normalization_486_betaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpBassignvariableop_32_gcn_attention_30_core_gcn_60_conv1d_243_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp@assignvariableop_33_gcn_attention_30_core_gcn_60_conv1d_243_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpNassignvariableop_34_gcn_attention_30_core_gcn_60_batch_normalization_487_gammaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpMassignvariableop_35_gcn_attention_30_core_gcn_60_batch_normalization_487_betaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpNassignvariableop_36_gcn_attention_30_core_gcn_60_batch_normalization_488_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpMassignvariableop_37_gcn_attention_30_core_gcn_60_batch_normalization_488_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpNassignvariableop_38_gcn_attention_30_core_gcn_60_batch_normalization_489_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpMassignvariableop_39_gcn_attention_30_core_gcn_60_batch_normalization_489_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpBassignvariableop_40_gcn_attention_30_core_gcn_61_conv2d_123_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp@assignvariableop_41_gcn_attention_30_core_gcn_61_conv2d_123_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpNassignvariableop_42_gcn_attention_30_core_gcn_61_batch_normalization_490_gammaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpMassignvariableop_43_gcn_attention_30_core_gcn_61_batch_normalization_490_betaIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpBassignvariableop_44_gcn_attention_30_core_gcn_61_conv1d_244_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp@assignvariableop_45_gcn_attention_30_core_gcn_61_conv1d_244_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpNassignvariableop_46_gcn_attention_30_core_gcn_61_batch_normalization_491_gammaIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpMassignvariableop_47_gcn_attention_30_core_gcn_61_batch_normalization_491_betaIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpBassignvariableop_48_gcn_attention_30_core_gcn_61_conv1d_245_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp@assignvariableop_49_gcn_attention_30_core_gcn_61_conv1d_245_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpNassignvariableop_50_gcn_attention_30_core_gcn_61_batch_normalization_492_gammaIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpMassignvariableop_51_gcn_attention_30_core_gcn_61_batch_normalization_492_betaIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpBassignvariableop_52_gcn_attention_30_core_gcn_61_conv1d_246_kernelIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp@assignvariableop_53_gcn_attention_30_core_gcn_61_conv1d_246_biasIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOpNassignvariableop_54_gcn_attention_30_core_gcn_61_batch_normalization_493_gammaIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOpMassignvariableop_55_gcn_attention_30_core_gcn_61_batch_normalization_493_betaIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOpNassignvariableop_56_gcn_attention_30_core_gcn_61_batch_normalization_494_gammaIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOpMassignvariableop_57_gcn_attention_30_core_gcn_61_batch_normalization_494_betaIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOpNassignvariableop_58_gcn_attention_30_core_gcn_61_batch_normalization_495_gammaIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOpMassignvariableop_59_gcn_attention_30_core_gcn_61_batch_normalization_495_betaIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOpTassignvariableop_60_gcn_attention_30_core_gcn_60_batch_normalization_484_moving_meanIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOpXassignvariableop_61_gcn_attention_30_core_gcn_60_batch_normalization_484_moving_varianceIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOpTassignvariableop_62_gcn_attention_30_core_gcn_60_batch_normalization_485_moving_meanIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOpXassignvariableop_63_gcn_attention_30_core_gcn_60_batch_normalization_485_moving_varianceIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOpTassignvariableop_64_gcn_attention_30_core_gcn_60_batch_normalization_486_moving_meanIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOpXassignvariableop_65_gcn_attention_30_core_gcn_60_batch_normalization_486_moving_varianceIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOpTassignvariableop_66_gcn_attention_30_core_gcn_60_batch_normalization_487_moving_meanIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOpXassignvariableop_67_gcn_attention_30_core_gcn_60_batch_normalization_487_moving_varianceIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOpTassignvariableop_68_gcn_attention_30_core_gcn_60_batch_normalization_488_moving_meanIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOpXassignvariableop_69_gcn_attention_30_core_gcn_60_batch_normalization_488_moving_varianceIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOpTassignvariableop_70_gcn_attention_30_core_gcn_60_batch_normalization_489_moving_meanIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOpXassignvariableop_71_gcn_attention_30_core_gcn_60_batch_normalization_489_moving_varianceIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOpTassignvariableop_72_gcn_attention_30_core_gcn_61_batch_normalization_490_moving_meanIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOpXassignvariableop_73_gcn_attention_30_core_gcn_61_batch_normalization_490_moving_varianceIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOpTassignvariableop_74_gcn_attention_30_core_gcn_61_batch_normalization_491_moving_meanIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOpXassignvariableop_75_gcn_attention_30_core_gcn_61_batch_normalization_491_moving_varianceIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOpTassignvariableop_76_gcn_attention_30_core_gcn_61_batch_normalization_492_moving_meanIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOpXassignvariableop_77_gcn_attention_30_core_gcn_61_batch_normalization_492_moving_varianceIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOpTassignvariableop_78_gcn_attention_30_core_gcn_61_batch_normalization_493_moving_meanIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOpXassignvariableop_79_gcn_attention_30_core_gcn_61_batch_normalization_493_moving_varianceIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOpTassignvariableop_80_gcn_attention_30_core_gcn_61_batch_normalization_494_moving_meanIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOpXassignvariableop_81_gcn_attention_30_core_gcn_61_batch_normalization_494_moving_varianceIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOpTassignvariableop_82_gcn_attention_30_core_gcn_61_batch_normalization_495_moving_meanIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOpXassignvariableop_83_gcn_attention_30_core_gcn_61_batch_normalization_495_moving_varianceIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_839
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_84Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_84f
Identity_85IdentityIdentity_84:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_85?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_85Identity_85:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
S__inference_batch_normalization_484_layer_call_and_return_conditional_losses_377640

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_489_layer_call_fn_380049

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_489_layer_call_and_return_conditional_losses_3783362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_489_layer_call_fn_380062

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_489_layer_call_and_return_conditional_losses_3783962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_491_layer_call_fn_380221

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_491_layer_call_and_return_conditional_losses_3786842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_484_layer_call_fn_379698

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_484_layer_call_and_return_conditional_losses_3776402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_492_layer_call_fn_380301

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_492_layer_call_and_return_conditional_losses_3788462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_489_layer_call_and_return_conditional_losses_378336

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_490_layer_call_fn_380159

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_490_layer_call_and_return_conditional_losses_3785402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_492_layer_call_fn_380288

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_492_layer_call_and_return_conditional_losses_3787862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?&
?
$__inference_signature_wrapper_377124
input_text1
input_text2
input_text3
unknown:?
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:	#
	unknown_6:		
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	

unknown_12:	$

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18: 

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24: 

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30: 

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:$

unknown_45:

unknown_46:

unknown_47:

unknown_48:

unknown_49:

unknown_50: 

unknown_51:

unknown_52:

unknown_53:

unknown_54:

unknown_55:

unknown_56: 

unknown_57:

unknown_58:

unknown_59:

unknown_60:

unknown_61:

unknown_62: 

unknown_63:

unknown_64:

unknown_65:

unknown_66:

unknown_67:

unknown_68:

unknown_69:

unknown_70:

unknown_71:

unknown_72:

unknown_73:

unknown_74:

unknown_75:

unknown_76:

unknown_77:

unknown_78:

unknown_79:

unknown_80: 

unknown_81:

unknown_82:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_text1input_text2input_text3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82*b
Tin[
Y2W*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*v
_read_only_resource_inputsX
VT	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUV*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference_tf_translate_3769472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????????????:+???????????????????????????:+???????????????????????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
5
_output_shapes#
!:???????????????????
%
_user_specified_nameinput_text1:nj
A
_output_shapes/
-:+???????????????????????????
%
_user_specified_nameinput_text2:nj
A
_output_shapes/
-:+???????????????????????????
%
_user_specified_nameinput_text3
?	
?
8__inference_batch_normalization_483_layer_call_fn_379565

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_483_layer_call_and_return_conditional_losses_3774362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_487_layer_call_and_return_conditional_losses_378048

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_379643d
Mgcn_attention_30_conv1d_240_kernel_regularizer_square_readvariableop_resource:?
identity??Dgcn_attention_30/conv1d_240/kernel/Regularizer/Square/ReadVariableOp?
Dgcn_attention_30/conv1d_240/kernel/Regularizer/Square/ReadVariableOpReadVariableOpMgcn_attention_30_conv1d_240_kernel_regularizer_square_readvariableop_resource*#
_output_shapes
:?*
dtype02F
Dgcn_attention_30/conv1d_240/kernel/Regularizer/Square/ReadVariableOp?
5gcn_attention_30/conv1d_240/kernel/Regularizer/SquareSquareLgcn_attention_30/conv1d_240/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:?27
5gcn_attention_30/conv1d_240/kernel/Regularizer/Square?
4gcn_attention_30/conv1d_240/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          26
4gcn_attention_30/conv1d_240/kernel/Regularizer/Const?
2gcn_attention_30/conv1d_240/kernel/Regularizer/SumSum9gcn_attention_30/conv1d_240/kernel/Regularizer/Square:y:0=gcn_attention_30/conv1d_240/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 24
2gcn_attention_30/conv1d_240/kernel/Regularizer/Sum?
4gcn_attention_30/conv1d_240/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??826
4gcn_attention_30/conv1d_240/kernel/Regularizer/mul/x?
2gcn_attention_30/conv1d_240/kernel/Regularizer/mulMul=gcn_attention_30/conv1d_240/kernel/Regularizer/mul/x:output:0;gcn_attention_30/conv1d_240/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 24
2gcn_attention_30/conv1d_240/kernel/Regularizer/mul?
IdentityIdentity6gcn_attention_30/conv1d_240/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOpE^gcn_attention_30/conv1d_240/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2?
Dgcn_attention_30/conv1d_240/kernel/Regularizer/Square/ReadVariableOpDgcn_attention_30/conv1d_240/kernel/Regularizer/Square/ReadVariableOp
?	
?
8__inference_batch_normalization_486_layer_call_fn_379840

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_batch_normalization_486_layer_call_and_return_conditional_losses_3779462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_491_layer_call_and_return_conditional_losses_378624

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
Q
input_text1B
serving_default_input_text1:0???????????????????
]
input_text2N
serving_default_input_text2:0+???????????????????????????
]
input_text3N
serving_default_input_text3:0+???????????????????????????H
outputs=
StatefulPartitionedCall:0??????????????????tensorflow/serving/predict:??
[
self_attention_layer

signatures
?tf_translate"
_generic_user_object
?
node0_Conv1D
node0_Activation
node0_BatchNormalization
node0_Dropout
edge_Convolution2D
edge_Activation
	edge_BatchNormalization

edge_Dropout
distance_Dense
distance_Convolution2D
distance_Activation
distance_BatchNormalization
distance_Dropout
	adj_Dense
node_Activation5
node_BatchNormalization5
node_Dropout5

gcn_layers
Activation_sig
node_Conv1D_out
relu_out
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
-
?serving_default"
signature_map
?

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
"	variables
#trainable_variables
$regularization_losses
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+	variables
,trainable_variables
-regularization_losses
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
/	variables
0trainable_variables
1regularization_losses
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
(
3	keras_api"
_tf_keras_layer
(
4	keras_api"
_tf_keras_layer
(
5	keras_api"
_tf_keras_layer
(
6	keras_api"
_tf_keras_layer
?

7kernel
8	variables
9trainable_variables
:regularization_losses
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Faxis
	Ggamma
Hbeta
Imoving_mean
Jmoving_variance
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
S
embeddings
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
\axis
	]gamma
^beta
_moving_mean
`moving_variance
a	variables
btrainable_variables
cregularization_losses
d	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
.
i0
j1"
trackable_list_wrapper
(
k	keras_api"
_tf_keras_layer
?

lkernel
mbias
n	variables
otrainable_variables
pregularization_losses
q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
r	variables
strainable_variables
tregularization_losses
u	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
0
1
'2
(3
74
<5
=6
G7
H8
S9
]10
^11
v12
w13
x14
y15
z16
{17
|18
}19
~20
21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
l52
m53
)54
*55
I56
J57
_58
`59
?60
?61
?62
?63
?64
?65
?66
?67
?68
?69
?70
?71
?72
?73
?74
?75
?76
?77
?78
?79
?80
?81
?82
?83"
trackable_list_wrapper
?
0
1
'2
(3
74
<5
=6
G7
H8
S9
]10
^11
v12
w13
x14
y15
z16
{17
|18
}19
~20
21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
l52
m53"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
	variables
?non_trainable_variables
trainable_variables
?layers
regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
9:7?2"gcn_attention_30/conv1d_240/kernel
.:,2 gcn_attention_30/conv1d_240/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
	variables
?non_trainable_variables
trainable_variables
?layers
 regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
"	variables
?non_trainable_variables
#trainable_variables
?layers
$regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<::2.gcn_attention_30/batch_normalization_480/gamma
;:92-gcn_attention_30/batch_normalization_480/beta
D:B (24gcn_attention_30/batch_normalization_480/moving_mean
H:F (28gcn_attention_30/batch_normalization_480/moving_variance
<
'0
(1
)2
*3"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
+	variables
?non_trainable_variables
,trainable_variables
?layers
-regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
/	variables
?non_trainable_variables
0trainable_variables
?layers
1regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
2:0	2 gcn_attention_30/dense_30/kernel
'
70"
trackable_list_wrapper
'
70"
trackable_list_wrapper
 "
trackable_list_wrapper
?
8	variables
?non_trainable_variables
9trainable_variables
?layers
:regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
<::		2"gcn_attention_30/conv2d_121/kernel
.:,	2 gcn_attention_30/conv2d_121/bias
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
>	variables
?non_trainable_variables
?trainable_variables
?layers
@regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
B	variables
?non_trainable_variables
Ctrainable_variables
?layers
Dregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<::	2.gcn_attention_30/batch_normalization_482/gamma
;:9	2-gcn_attention_30/batch_normalization_482/beta
D:B	 (24gcn_attention_30/batch_normalization_482/moving_mean
H:F	 (28gcn_attention_30/batch_normalization_482/moving_variance
<
G0
H1
I2
J3"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
K	variables
?non_trainable_variables
Ltrainable_variables
?layers
Mregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
O	variables
?non_trainable_variables
Ptrainable_variables
?layers
Qregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
::8	2(gcn_attention_30/embedding_30/embeddings
'
S0"
trackable_list_wrapper
'
S0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
T	variables
?non_trainable_variables
Utrainable_variables
?layers
Vregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
X	variables
?non_trainable_variables
Ytrainable_variables
?layers
Zregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<::2.gcn_attention_30/batch_normalization_483/gamma
;:92-gcn_attention_30/batch_normalization_483/beta
D:B (24gcn_attention_30/batch_normalization_483/moving_mean
H:F (28gcn_attention_30/batch_normalization_483/moving_variance
<
]0
^1
_2
`3"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
a	variables
?non_trainable_variables
btrainable_variables
?layers
cregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
e	variables
?non_trainable_variables
ftrainable_variables
?layers
gregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?conv2d_1
?batch_norm_1
?	dropout_1
?conv1d_1
?batch_norm_2
?	dropout_2
?conv1d_2
?batch_norm_3
?	dropout_3
?conv1d_3
?batch_norm_4
?	dropout_4
?batch_norm_5
?batch_norm_6
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?conv2d_1
?batch_norm_1
?	dropout_1
?conv1d_1
?batch_norm_2
?	dropout_2
?conv1d_2
?batch_norm_3
?	dropout_3
?conv1d_3
?batch_norm_4
?	dropout_4
?batch_norm_5
?batch_norm_6
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_generic_user_object
8:62"gcn_attention_30/conv1d_247/kernel
.:,2 gcn_attention_30/conv1d_247/bias
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
n	variables
?non_trainable_variables
otrainable_variables
?layers
pregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
r	variables
?non_trainable_variables
strainable_variables
?layers
tregularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
H:F2.gcn_attention_30/core_gcn_60/conv2d_122/kernel
::82,gcn_attention_30/core_gcn_60/conv2d_122/bias
H:F2:gcn_attention_30/core_gcn_60/batch_normalization_484/gamma
G:E29gcn_attention_30/core_gcn_60/batch_normalization_484/beta
D:B2.gcn_attention_30/core_gcn_60/conv1d_241/kernel
::82,gcn_attention_30/core_gcn_60/conv1d_241/bias
H:F2:gcn_attention_30/core_gcn_60/batch_normalization_485/gamma
G:E29gcn_attention_30/core_gcn_60/batch_normalization_485/beta
D:B2.gcn_attention_30/core_gcn_60/conv1d_242/kernel
::82,gcn_attention_30/core_gcn_60/conv1d_242/bias
H:F2:gcn_attention_30/core_gcn_60/batch_normalization_486/gamma
G:E29gcn_attention_30/core_gcn_60/batch_normalization_486/beta
D:B2.gcn_attention_30/core_gcn_60/conv1d_243/kernel
::82,gcn_attention_30/core_gcn_60/conv1d_243/bias
H:F2:gcn_attention_30/core_gcn_60/batch_normalization_487/gamma
G:E29gcn_attention_30/core_gcn_60/batch_normalization_487/beta
H:F2:gcn_attention_30/core_gcn_60/batch_normalization_488/gamma
G:E29gcn_attention_30/core_gcn_60/batch_normalization_488/beta
H:F2:gcn_attention_30/core_gcn_60/batch_normalization_489/gamma
G:E29gcn_attention_30/core_gcn_60/batch_normalization_489/beta
H:F2.gcn_attention_30/core_gcn_61/conv2d_123/kernel
::82,gcn_attention_30/core_gcn_61/conv2d_123/bias
H:F2:gcn_attention_30/core_gcn_61/batch_normalization_490/gamma
G:E29gcn_attention_30/core_gcn_61/batch_normalization_490/beta
D:B2.gcn_attention_30/core_gcn_61/conv1d_244/kernel
::82,gcn_attention_30/core_gcn_61/conv1d_244/bias
H:F2:gcn_attention_30/core_gcn_61/batch_normalization_491/gamma
G:E29gcn_attention_30/core_gcn_61/batch_normalization_491/beta
D:B2.gcn_attention_30/core_gcn_61/conv1d_245/kernel
::82,gcn_attention_30/core_gcn_61/conv1d_245/bias
H:F2:gcn_attention_30/core_gcn_61/batch_normalization_492/gamma
G:E29gcn_attention_30/core_gcn_61/batch_normalization_492/beta
D:B2.gcn_attention_30/core_gcn_61/conv1d_246/kernel
::82,gcn_attention_30/core_gcn_61/conv1d_246/bias
H:F2:gcn_attention_30/core_gcn_61/batch_normalization_493/gamma
G:E29gcn_attention_30/core_gcn_61/batch_normalization_493/beta
H:F2:gcn_attention_30/core_gcn_61/batch_normalization_494/gamma
G:E29gcn_attention_30/core_gcn_61/batch_normalization_494/beta
H:F2:gcn_attention_30/core_gcn_61/batch_normalization_495/gamma
G:E29gcn_attention_30/core_gcn_61/batch_normalization_495/beta
P:N (2@gcn_attention_30/core_gcn_60/batch_normalization_484/moving_mean
T:R (2Dgcn_attention_30/core_gcn_60/batch_normalization_484/moving_variance
P:N (2@gcn_attention_30/core_gcn_60/batch_normalization_485/moving_mean
T:R (2Dgcn_attention_30/core_gcn_60/batch_normalization_485/moving_variance
P:N (2@gcn_attention_30/core_gcn_60/batch_normalization_486/moving_mean
T:R (2Dgcn_attention_30/core_gcn_60/batch_normalization_486/moving_variance
P:N (2@gcn_attention_30/core_gcn_60/batch_normalization_487/moving_mean
T:R (2Dgcn_attention_30/core_gcn_60/batch_normalization_487/moving_variance
P:N (2@gcn_attention_30/core_gcn_60/batch_normalization_488/moving_mean
T:R (2Dgcn_attention_30/core_gcn_60/batch_normalization_488/moving_variance
P:N (2@gcn_attention_30/core_gcn_60/batch_normalization_489/moving_mean
T:R (2Dgcn_attention_30/core_gcn_60/batch_normalization_489/moving_variance
P:N (2@gcn_attention_30/core_gcn_61/batch_normalization_490/moving_mean
T:R (2Dgcn_attention_30/core_gcn_61/batch_normalization_490/moving_variance
P:N (2@gcn_attention_30/core_gcn_61/batch_normalization_491/moving_mean
T:R (2Dgcn_attention_30/core_gcn_61/batch_normalization_491/moving_variance
P:N (2@gcn_attention_30/core_gcn_61/batch_normalization_492/moving_mean
T:R (2Dgcn_attention_30/core_gcn_61/batch_normalization_492/moving_variance
P:N (2@gcn_attention_30/core_gcn_61/batch_normalization_493/moving_mean
T:R (2Dgcn_attention_30/core_gcn_61/batch_normalization_493/moving_variance
P:N (2@gcn_attention_30/core_gcn_61/batch_normalization_494/moving_mean
T:R (2Dgcn_attention_30/core_gcn_61/batch_normalization_494/moving_variance
P:N (2@gcn_attention_30/core_gcn_61/batch_normalization_495/moving_mean
T:R (2Dgcn_attention_30/core_gcn_61/batch_normalization_495/moving_variance
?
)0
*1
I2
J3
_4
`5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29"
trackable_list_wrapper
?
0
1
2
3
4
5
	6

7
8
9
10
11
12
13
14
15
16
i17
j18
19
20
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?

vkernel
wbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	xgamma
ybeta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

zkernel
{bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	|gamma
}beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

~kernel
bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
v0
w1
x2
y3
z4
{5
|6
}7
~8
9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31"
trackable_list_wrapper
?
v0
w1
x2
y3
z4
{5
|6
}7
~8
9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31"
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
>
x0
y1
?2
?3"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
>
|0
}1
?2
?3"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11"
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?non_trainable_variables
?trainable_variables
?layers
?regularization_losses
?metrics
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11"
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?2?
__inference_tf_translate_376947?
???
FullArgSpec@
args8?5
jself
jinput_text1
jinput_text2
jinput_text3
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *???
&?#???????????????????
2?/+???????????????????????????
2?/+???????????????????????????
?2??
???
FullArgSpecd
args\?Y
jself
jbatch_target_feat
jseq_contact_batch
jpdb_distance_pair_batch

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpecd
args\?Y
jself
jbatch_target_feat
jseq_contact_batch
jpdb_distance_pair_batch

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_377124input_text1input_text2input_text3"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_batch_normalization_480_layer_call_fn_379417
8__inference_batch_normalization_480_layer_call_fn_379430?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_480_layer_call_and_return_conditional_losses_379450
S__inference_batch_normalization_480_layer_call_and_return_conditional_losses_379484?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_batch_normalization_482_layer_call_fn_379503
8__inference_batch_normalization_482_layer_call_fn_379516?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_482_layer_call_and_return_conditional_losses_379534
S__inference_batch_normalization_482_layer_call_and_return_conditional_losses_379552?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_batch_normalization_483_layer_call_fn_379565
8__inference_batch_normalization_483_layer_call_fn_379578?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_483_layer_call_and_return_conditional_losses_379598
S__inference_batch_normalization_483_layer_call_and_return_conditional_losses_379632?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_379643?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_379654?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2??
???
FullArgSpec/
args'?$
jself
jx_in
je_in

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec/
args'?$
jself
jx_in
je_in

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec/
args'?$
jself
jx_in
je_in

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec/
args'?$
jself
jx_in
je_in

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_batch_normalization_484_layer_call_fn_379685
8__inference_batch_normalization_484_layer_call_fn_379698?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_484_layer_call_and_return_conditional_losses_379716
S__inference_batch_normalization_484_layer_call_and_return_conditional_losses_379734?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_batch_normalization_485_layer_call_fn_379747
8__inference_batch_normalization_485_layer_call_fn_379760?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_485_layer_call_and_return_conditional_losses_379780
S__inference_batch_normalization_485_layer_call_and_return_conditional_losses_379814?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_batch_normalization_486_layer_call_fn_379827
8__inference_batch_normalization_486_layer_call_fn_379840?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_486_layer_call_and_return_conditional_losses_379860
S__inference_batch_normalization_486_layer_call_and_return_conditional_losses_379894?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_batch_normalization_487_layer_call_fn_379907
8__inference_batch_normalization_487_layer_call_fn_379920?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_487_layer_call_and_return_conditional_losses_379940
S__inference_batch_normalization_487_layer_call_and_return_conditional_losses_379974?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_488_layer_call_fn_379987
8__inference_batch_normalization_488_layer_call_fn_380000?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_488_layer_call_and_return_conditional_losses_380018
S__inference_batch_normalization_488_layer_call_and_return_conditional_losses_380036?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_489_layer_call_fn_380049
8__inference_batch_normalization_489_layer_call_fn_380062?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_489_layer_call_and_return_conditional_losses_380082
S__inference_batch_normalization_489_layer_call_and_return_conditional_losses_380116?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference_loss_fn_2_380127?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_batch_normalization_490_layer_call_fn_380146
8__inference_batch_normalization_490_layer_call_fn_380159?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_490_layer_call_and_return_conditional_losses_380177
S__inference_batch_normalization_490_layer_call_and_return_conditional_losses_380195?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_batch_normalization_491_layer_call_fn_380208
8__inference_batch_normalization_491_layer_call_fn_380221?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_491_layer_call_and_return_conditional_losses_380241
S__inference_batch_normalization_491_layer_call_and_return_conditional_losses_380275?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_batch_normalization_492_layer_call_fn_380288
8__inference_batch_normalization_492_layer_call_fn_380301?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_492_layer_call_and_return_conditional_losses_380321
S__inference_batch_normalization_492_layer_call_and_return_conditional_losses_380355?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_batch_normalization_493_layer_call_fn_380368
8__inference_batch_normalization_493_layer_call_fn_380381?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_493_layer_call_and_return_conditional_losses_380401
S__inference_batch_normalization_493_layer_call_and_return_conditional_losses_380435?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_494_layer_call_fn_380448
8__inference_batch_normalization_494_layer_call_fn_380461?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_494_layer_call_and_return_conditional_losses_380479
S__inference_batch_normalization_494_layer_call_and_return_conditional_losses_380497?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_batch_normalization_495_layer_call_fn_380510
8__inference_batch_normalization_495_layer_call_fn_380523?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_495_layer_call_and_return_conditional_losses_380543
S__inference_batch_normalization_495_layer_call_and_return_conditional_losses_380577?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference_loss_fn_3_380588?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? ?
S__inference_batch_normalization_480_layer_call_and_return_conditional_losses_379450|)*('@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
S__inference_batch_normalization_480_layer_call_and_return_conditional_losses_379484|)*('@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
8__inference_batch_normalization_480_layer_call_fn_379417o)*('@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
8__inference_batch_normalization_480_layer_call_fn_379430o)*('@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
S__inference_batch_normalization_482_layer_call_and_return_conditional_losses_379534?GHIJM?J
C?@
:?7
inputs+???????????????????????????	
p 
? "??<
5?2
0+???????????????????????????	
? ?
S__inference_batch_normalization_482_layer_call_and_return_conditional_losses_379552?GHIJM?J
C?@
:?7
inputs+???????????????????????????	
p
? "??<
5?2
0+???????????????????????????	
? ?
8__inference_batch_normalization_482_layer_call_fn_379503?GHIJM?J
C?@
:?7
inputs+???????????????????????????	
p 
? "2?/+???????????????????????????	?
8__inference_batch_normalization_482_layer_call_fn_379516?GHIJM?J
C?@
:?7
inputs+???????????????????????????	
p
? "2?/+???????????????????????????	?
S__inference_batch_normalization_483_layer_call_and_return_conditional_losses_379598|_`^]@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
S__inference_batch_normalization_483_layer_call_and_return_conditional_losses_379632|_`^]@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
8__inference_batch_normalization_483_layer_call_fn_379565o_`^]@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
8__inference_batch_normalization_483_layer_call_fn_379578o_`^]@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
S__inference_batch_normalization_484_layer_call_and_return_conditional_losses_379716?xy??M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_484_layer_call_and_return_conditional_losses_379734?xy??M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_484_layer_call_fn_379685?xy??M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_484_layer_call_fn_379698?xy??M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
S__inference_batch_normalization_485_layer_call_and_return_conditional_losses_379780~??}|@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
S__inference_batch_normalization_485_layer_call_and_return_conditional_losses_379814~??}|@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
8__inference_batch_normalization_485_layer_call_fn_379747q??}|@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
8__inference_batch_normalization_485_layer_call_fn_379760q??}|@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
S__inference_batch_normalization_486_layer_call_and_return_conditional_losses_379860?????@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
S__inference_batch_normalization_486_layer_call_and_return_conditional_losses_379894?????@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
8__inference_batch_normalization_486_layer_call_fn_379827s????@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
8__inference_batch_normalization_486_layer_call_fn_379840s????@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
S__inference_batch_normalization_487_layer_call_and_return_conditional_losses_379940?????@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
S__inference_batch_normalization_487_layer_call_and_return_conditional_losses_379974?????@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
8__inference_batch_normalization_487_layer_call_fn_379907s????@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
8__inference_batch_normalization_487_layer_call_fn_379920s????@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
S__inference_batch_normalization_488_layer_call_and_return_conditional_losses_380018?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_488_layer_call_and_return_conditional_losses_380036?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_488_layer_call_fn_379987?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_488_layer_call_fn_380000?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
S__inference_batch_normalization_489_layer_call_and_return_conditional_losses_380082?????@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
S__inference_batch_normalization_489_layer_call_and_return_conditional_losses_380116?????@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
8__inference_batch_normalization_489_layer_call_fn_380049s????@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
8__inference_batch_normalization_489_layer_call_fn_380062s????@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
S__inference_batch_normalization_490_layer_call_and_return_conditional_losses_380177?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_490_layer_call_and_return_conditional_losses_380195?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_490_layer_call_fn_380146?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_490_layer_call_fn_380159?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
S__inference_batch_normalization_491_layer_call_and_return_conditional_losses_380241?????@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
S__inference_batch_normalization_491_layer_call_and_return_conditional_losses_380275?????@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
8__inference_batch_normalization_491_layer_call_fn_380208s????@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
8__inference_batch_normalization_491_layer_call_fn_380221s????@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
S__inference_batch_normalization_492_layer_call_and_return_conditional_losses_380321?????@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
S__inference_batch_normalization_492_layer_call_and_return_conditional_losses_380355?????@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
8__inference_batch_normalization_492_layer_call_fn_380288s????@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
8__inference_batch_normalization_492_layer_call_fn_380301s????@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
S__inference_batch_normalization_493_layer_call_and_return_conditional_losses_380401?????@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
S__inference_batch_normalization_493_layer_call_and_return_conditional_losses_380435?????@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
8__inference_batch_normalization_493_layer_call_fn_380368s????@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
8__inference_batch_normalization_493_layer_call_fn_380381s????@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
S__inference_batch_normalization_494_layer_call_and_return_conditional_losses_380479?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_494_layer_call_and_return_conditional_losses_380497?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_494_layer_call_fn_380448?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_494_layer_call_fn_380461?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
S__inference_batch_normalization_495_layer_call_and_return_conditional_losses_380543?????@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
S__inference_batch_normalization_495_layer_call_and_return_conditional_losses_380577?????@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
8__inference_batch_normalization_495_layer_call_fn_380510s????@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
8__inference_batch_normalization_495_layer_call_fn_380523s????@?=
6?3
-?*
inputs??????????????????
p
? "%?"??????????????????;
__inference_loss_fn_0_379643?

? 
? "? ;
__inference_loss_fn_1_379654<?

? 
? "? ;
__inference_loss_fn_2_380127v?

? 
? "? <
__inference_loss_fn_3_380588??

? 
? "? ?
$__inference_signature_wrapper_377124??)*('7<=GHIJSvwxy??z{??}|~??????????????????????????????????????????????????_`^]lm???
? 
???
B
input_text13?0
input_text1???????????????????
N
input_text2??<
input_text2+???????????????????????????
N
input_text3??<
input_text3+???????????????????????????">?;
9
outputs.?+
outputs???????????????????
__inference_tf_translate_376947??)*('7<=GHIJSvwxy??z{??}|~??????????????????????????????????????????????????_`^]lm???
???
3?0
input_text1???????????????????
??<
input_text2+???????????????????????????
??<
input_text3+???????????????????????????
? ">?;
9
outputs.?+
outputs??????????????????