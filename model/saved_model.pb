ЭР-
÷%ђ%
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
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
dtypetypeИ
†
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
Ы
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
ъ
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
epsilonfloat%Ј—8"&
exponential_avg_factorfloat%  А?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
≠
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
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(Р
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
Н
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
dtypetypeИ
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
•
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
list(type)(0И
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
©
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
Њ
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
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
ц
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
М
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.6.22v2.6.1-9-gc2363d6d0258Е–'
•
"gcn_attention_49/conv1d_392/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"gcn_attention_49/conv1d_392/kernel
Ю
6gcn_attention_49/conv1d_392/kernel/Read/ReadVariableOpReadVariableOp"gcn_attention_49/conv1d_392/kernel*#
_output_shapes
:ђ*
dtype0
Ш
 gcn_attention_49/conv1d_392/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" gcn_attention_49/conv1d_392/bias
С
4gcn_attention_49/conv1d_392/bias/Read/ReadVariableOpReadVariableOp gcn_attention_49/conv1d_392/bias*
_output_shapes
:*
dtype0
і
.gcn_attention_49/batch_normalization_784/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.gcn_attention_49/batch_normalization_784/gamma
≠
Bgcn_attention_49/batch_normalization_784/gamma/Read/ReadVariableOpReadVariableOp.gcn_attention_49/batch_normalization_784/gamma*
_output_shapes
:*
dtype0
≤
-gcn_attention_49/batch_normalization_784/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-gcn_attention_49/batch_normalization_784/beta
Ђ
Agcn_attention_49/batch_normalization_784/beta/Read/ReadVariableOpReadVariableOp-gcn_attention_49/batch_normalization_784/beta*
_output_shapes
:*
dtype0
ј
4gcn_attention_49/batch_normalization_784/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64gcn_attention_49/batch_normalization_784/moving_mean
є
Hgcn_attention_49/batch_normalization_784/moving_mean/Read/ReadVariableOpReadVariableOp4gcn_attention_49/batch_normalization_784/moving_mean*
_output_shapes
:*
dtype0
»
8gcn_attention_49/batch_normalization_784/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8gcn_attention_49/batch_normalization_784/moving_variance
Ѕ
Lgcn_attention_49/batch_normalization_784/moving_variance/Read/ReadVariableOpReadVariableOp8gcn_attention_49/batch_normalization_784/moving_variance*
_output_shapes
:*
dtype0
Ь
 gcn_attention_49/dense_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*1
shared_name" gcn_attention_49/dense_49/kernel
Х
4gcn_attention_49/dense_49/kernel/Read/ReadVariableOpReadVariableOp gcn_attention_49/dense_49/kernel*
_output_shapes

:	*
dtype0
®
"gcn_attention_49/conv2d_197/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*3
shared_name$"gcn_attention_49/conv2d_197/kernel
°
6gcn_attention_49/conv2d_197/kernel/Read/ReadVariableOpReadVariableOp"gcn_attention_49/conv2d_197/kernel*&
_output_shapes
:		*
dtype0
Ш
 gcn_attention_49/conv2d_197/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" gcn_attention_49/conv2d_197/bias
С
4gcn_attention_49/conv2d_197/bias/Read/ReadVariableOpReadVariableOp gcn_attention_49/conv2d_197/bias*
_output_shapes
:	*
dtype0
і
.gcn_attention_49/batch_normalization_786/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*?
shared_name0.gcn_attention_49/batch_normalization_786/gamma
≠
Bgcn_attention_49/batch_normalization_786/gamma/Read/ReadVariableOpReadVariableOp.gcn_attention_49/batch_normalization_786/gamma*
_output_shapes
:	*
dtype0
≤
-gcn_attention_49/batch_normalization_786/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*>
shared_name/-gcn_attention_49/batch_normalization_786/beta
Ђ
Agcn_attention_49/batch_normalization_786/beta/Read/ReadVariableOpReadVariableOp-gcn_attention_49/batch_normalization_786/beta*
_output_shapes
:	*
dtype0
ј
4gcn_attention_49/batch_normalization_786/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*E
shared_name64gcn_attention_49/batch_normalization_786/moving_mean
є
Hgcn_attention_49/batch_normalization_786/moving_mean/Read/ReadVariableOpReadVariableOp4gcn_attention_49/batch_normalization_786/moving_mean*
_output_shapes
:	*
dtype0
»
8gcn_attention_49/batch_normalization_786/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8gcn_attention_49/batch_normalization_786/moving_variance
Ѕ
Lgcn_attention_49/batch_normalization_786/moving_variance/Read/ReadVariableOpReadVariableOp8gcn_attention_49/batch_normalization_786/moving_variance*
_output_shapes
:	*
dtype0
ђ
(gcn_attention_49/embedding_49/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*9
shared_name*(gcn_attention_49/embedding_49/embeddings
•
<gcn_attention_49/embedding_49/embeddings/Read/ReadVariableOpReadVariableOp(gcn_attention_49/embedding_49/embeddings*
_output_shapes

:	*
dtype0
і
.gcn_attention_49/batch_normalization_787/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.gcn_attention_49/batch_normalization_787/gamma
≠
Bgcn_attention_49/batch_normalization_787/gamma/Read/ReadVariableOpReadVariableOp.gcn_attention_49/batch_normalization_787/gamma*
_output_shapes
:*
dtype0
≤
-gcn_attention_49/batch_normalization_787/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-gcn_attention_49/batch_normalization_787/beta
Ђ
Agcn_attention_49/batch_normalization_787/beta/Read/ReadVariableOpReadVariableOp-gcn_attention_49/batch_normalization_787/beta*
_output_shapes
:*
dtype0
ј
4gcn_attention_49/batch_normalization_787/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64gcn_attention_49/batch_normalization_787/moving_mean
є
Hgcn_attention_49/batch_normalization_787/moving_mean/Read/ReadVariableOpReadVariableOp4gcn_attention_49/batch_normalization_787/moving_mean*
_output_shapes
:*
dtype0
»
8gcn_attention_49/batch_normalization_787/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8gcn_attention_49/batch_normalization_787/moving_variance
Ѕ
Lgcn_attention_49/batch_normalization_787/moving_variance/Read/ReadVariableOpReadVariableOp8gcn_attention_49/batch_normalization_787/moving_variance*
_output_shapes
:*
dtype0
§
"gcn_attention_49/conv1d_399/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"gcn_attention_49/conv1d_399/kernel
Э
6gcn_attention_49/conv1d_399/kernel/Read/ReadVariableOpReadVariableOp"gcn_attention_49/conv1d_399/kernel*"
_output_shapes
:*
dtype0
Ш
 gcn_attention_49/conv1d_399/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" gcn_attention_49/conv1d_399/bias
С
4gcn_attention_49/conv1d_399/bias/Read/ReadVariableOpReadVariableOp gcn_attention_49/conv1d_399/bias*
_output_shapes
:*
dtype0
ј
.gcn_attention_49/core_gcn_98/conv2d_198/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.gcn_attention_49/core_gcn_98/conv2d_198/kernel
є
Bgcn_attention_49/core_gcn_98/conv2d_198/kernel/Read/ReadVariableOpReadVariableOp.gcn_attention_49/core_gcn_98/conv2d_198/kernel*&
_output_shapes
:*
dtype0
∞
,gcn_attention_49/core_gcn_98/conv2d_198/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,gcn_attention_49/core_gcn_98/conv2d_198/bias
©
@gcn_attention_49/core_gcn_98/conv2d_198/bias/Read/ReadVariableOpReadVariableOp,gcn_attention_49/core_gcn_98/conv2d_198/bias*
_output_shapes
:*
dtype0
ћ
:gcn_attention_49/core_gcn_98/batch_normalization_788/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_49/core_gcn_98/batch_normalization_788/gamma
≈
Ngcn_attention_49/core_gcn_98/batch_normalization_788/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_49/core_gcn_98/batch_normalization_788/gamma*
_output_shapes
:*
dtype0
 
9gcn_attention_49/core_gcn_98/batch_normalization_788/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_49/core_gcn_98/batch_normalization_788/beta
√
Mgcn_attention_49/core_gcn_98/batch_normalization_788/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_49/core_gcn_98/batch_normalization_788/beta*
_output_shapes
:*
dtype0
Љ
.gcn_attention_49/core_gcn_98/conv1d_393/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.gcn_attention_49/core_gcn_98/conv1d_393/kernel
µ
Bgcn_attention_49/core_gcn_98/conv1d_393/kernel/Read/ReadVariableOpReadVariableOp.gcn_attention_49/core_gcn_98/conv1d_393/kernel*"
_output_shapes
:*
dtype0
∞
,gcn_attention_49/core_gcn_98/conv1d_393/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,gcn_attention_49/core_gcn_98/conv1d_393/bias
©
@gcn_attention_49/core_gcn_98/conv1d_393/bias/Read/ReadVariableOpReadVariableOp,gcn_attention_49/core_gcn_98/conv1d_393/bias*
_output_shapes
:*
dtype0
ћ
:gcn_attention_49/core_gcn_98/batch_normalization_789/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_49/core_gcn_98/batch_normalization_789/gamma
≈
Ngcn_attention_49/core_gcn_98/batch_normalization_789/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_49/core_gcn_98/batch_normalization_789/gamma*
_output_shapes
:*
dtype0
 
9gcn_attention_49/core_gcn_98/batch_normalization_789/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_49/core_gcn_98/batch_normalization_789/beta
√
Mgcn_attention_49/core_gcn_98/batch_normalization_789/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_49/core_gcn_98/batch_normalization_789/beta*
_output_shapes
:*
dtype0
Љ
.gcn_attention_49/core_gcn_98/conv1d_394/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.gcn_attention_49/core_gcn_98/conv1d_394/kernel
µ
Bgcn_attention_49/core_gcn_98/conv1d_394/kernel/Read/ReadVariableOpReadVariableOp.gcn_attention_49/core_gcn_98/conv1d_394/kernel*"
_output_shapes
:*
dtype0
∞
,gcn_attention_49/core_gcn_98/conv1d_394/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,gcn_attention_49/core_gcn_98/conv1d_394/bias
©
@gcn_attention_49/core_gcn_98/conv1d_394/bias/Read/ReadVariableOpReadVariableOp,gcn_attention_49/core_gcn_98/conv1d_394/bias*
_output_shapes
:*
dtype0
ћ
:gcn_attention_49/core_gcn_98/batch_normalization_790/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_49/core_gcn_98/batch_normalization_790/gamma
≈
Ngcn_attention_49/core_gcn_98/batch_normalization_790/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_49/core_gcn_98/batch_normalization_790/gamma*
_output_shapes
:*
dtype0
 
9gcn_attention_49/core_gcn_98/batch_normalization_790/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_49/core_gcn_98/batch_normalization_790/beta
√
Mgcn_attention_49/core_gcn_98/batch_normalization_790/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_49/core_gcn_98/batch_normalization_790/beta*
_output_shapes
:*
dtype0
Љ
.gcn_attention_49/core_gcn_98/conv1d_395/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.gcn_attention_49/core_gcn_98/conv1d_395/kernel
µ
Bgcn_attention_49/core_gcn_98/conv1d_395/kernel/Read/ReadVariableOpReadVariableOp.gcn_attention_49/core_gcn_98/conv1d_395/kernel*"
_output_shapes
:*
dtype0
∞
,gcn_attention_49/core_gcn_98/conv1d_395/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,gcn_attention_49/core_gcn_98/conv1d_395/bias
©
@gcn_attention_49/core_gcn_98/conv1d_395/bias/Read/ReadVariableOpReadVariableOp,gcn_attention_49/core_gcn_98/conv1d_395/bias*
_output_shapes
:*
dtype0
ћ
:gcn_attention_49/core_gcn_98/batch_normalization_791/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_49/core_gcn_98/batch_normalization_791/gamma
≈
Ngcn_attention_49/core_gcn_98/batch_normalization_791/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_49/core_gcn_98/batch_normalization_791/gamma*
_output_shapes
:*
dtype0
 
9gcn_attention_49/core_gcn_98/batch_normalization_791/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_49/core_gcn_98/batch_normalization_791/beta
√
Mgcn_attention_49/core_gcn_98/batch_normalization_791/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_49/core_gcn_98/batch_normalization_791/beta*
_output_shapes
:*
dtype0
ћ
:gcn_attention_49/core_gcn_98/batch_normalization_792/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_49/core_gcn_98/batch_normalization_792/gamma
≈
Ngcn_attention_49/core_gcn_98/batch_normalization_792/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_49/core_gcn_98/batch_normalization_792/gamma*
_output_shapes
:*
dtype0
 
9gcn_attention_49/core_gcn_98/batch_normalization_792/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_49/core_gcn_98/batch_normalization_792/beta
√
Mgcn_attention_49/core_gcn_98/batch_normalization_792/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_49/core_gcn_98/batch_normalization_792/beta*
_output_shapes
:*
dtype0
ћ
:gcn_attention_49/core_gcn_98/batch_normalization_793/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_49/core_gcn_98/batch_normalization_793/gamma
≈
Ngcn_attention_49/core_gcn_98/batch_normalization_793/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_49/core_gcn_98/batch_normalization_793/gamma*
_output_shapes
:*
dtype0
 
9gcn_attention_49/core_gcn_98/batch_normalization_793/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_49/core_gcn_98/batch_normalization_793/beta
√
Mgcn_attention_49/core_gcn_98/batch_normalization_793/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_49/core_gcn_98/batch_normalization_793/beta*
_output_shapes
:*
dtype0
ј
.gcn_attention_49/core_gcn_99/conv2d_199/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.gcn_attention_49/core_gcn_99/conv2d_199/kernel
є
Bgcn_attention_49/core_gcn_99/conv2d_199/kernel/Read/ReadVariableOpReadVariableOp.gcn_attention_49/core_gcn_99/conv2d_199/kernel*&
_output_shapes
:*
dtype0
∞
,gcn_attention_49/core_gcn_99/conv2d_199/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,gcn_attention_49/core_gcn_99/conv2d_199/bias
©
@gcn_attention_49/core_gcn_99/conv2d_199/bias/Read/ReadVariableOpReadVariableOp,gcn_attention_49/core_gcn_99/conv2d_199/bias*
_output_shapes
:*
dtype0
ћ
:gcn_attention_49/core_gcn_99/batch_normalization_794/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_49/core_gcn_99/batch_normalization_794/gamma
≈
Ngcn_attention_49/core_gcn_99/batch_normalization_794/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_49/core_gcn_99/batch_normalization_794/gamma*
_output_shapes
:*
dtype0
 
9gcn_attention_49/core_gcn_99/batch_normalization_794/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_49/core_gcn_99/batch_normalization_794/beta
√
Mgcn_attention_49/core_gcn_99/batch_normalization_794/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_49/core_gcn_99/batch_normalization_794/beta*
_output_shapes
:*
dtype0
Љ
.gcn_attention_49/core_gcn_99/conv1d_396/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.gcn_attention_49/core_gcn_99/conv1d_396/kernel
µ
Bgcn_attention_49/core_gcn_99/conv1d_396/kernel/Read/ReadVariableOpReadVariableOp.gcn_attention_49/core_gcn_99/conv1d_396/kernel*"
_output_shapes
:*
dtype0
∞
,gcn_attention_49/core_gcn_99/conv1d_396/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,gcn_attention_49/core_gcn_99/conv1d_396/bias
©
@gcn_attention_49/core_gcn_99/conv1d_396/bias/Read/ReadVariableOpReadVariableOp,gcn_attention_49/core_gcn_99/conv1d_396/bias*
_output_shapes
:*
dtype0
ћ
:gcn_attention_49/core_gcn_99/batch_normalization_795/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_49/core_gcn_99/batch_normalization_795/gamma
≈
Ngcn_attention_49/core_gcn_99/batch_normalization_795/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_49/core_gcn_99/batch_normalization_795/gamma*
_output_shapes
:*
dtype0
 
9gcn_attention_49/core_gcn_99/batch_normalization_795/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_49/core_gcn_99/batch_normalization_795/beta
√
Mgcn_attention_49/core_gcn_99/batch_normalization_795/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_49/core_gcn_99/batch_normalization_795/beta*
_output_shapes
:*
dtype0
Љ
.gcn_attention_49/core_gcn_99/conv1d_397/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.gcn_attention_49/core_gcn_99/conv1d_397/kernel
µ
Bgcn_attention_49/core_gcn_99/conv1d_397/kernel/Read/ReadVariableOpReadVariableOp.gcn_attention_49/core_gcn_99/conv1d_397/kernel*"
_output_shapes
:*
dtype0
∞
,gcn_attention_49/core_gcn_99/conv1d_397/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,gcn_attention_49/core_gcn_99/conv1d_397/bias
©
@gcn_attention_49/core_gcn_99/conv1d_397/bias/Read/ReadVariableOpReadVariableOp,gcn_attention_49/core_gcn_99/conv1d_397/bias*
_output_shapes
:*
dtype0
ћ
:gcn_attention_49/core_gcn_99/batch_normalization_796/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_49/core_gcn_99/batch_normalization_796/gamma
≈
Ngcn_attention_49/core_gcn_99/batch_normalization_796/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_49/core_gcn_99/batch_normalization_796/gamma*
_output_shapes
:*
dtype0
 
9gcn_attention_49/core_gcn_99/batch_normalization_796/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_49/core_gcn_99/batch_normalization_796/beta
√
Mgcn_attention_49/core_gcn_99/batch_normalization_796/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_49/core_gcn_99/batch_normalization_796/beta*
_output_shapes
:*
dtype0
Љ
.gcn_attention_49/core_gcn_99/conv1d_398/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.gcn_attention_49/core_gcn_99/conv1d_398/kernel
µ
Bgcn_attention_49/core_gcn_99/conv1d_398/kernel/Read/ReadVariableOpReadVariableOp.gcn_attention_49/core_gcn_99/conv1d_398/kernel*"
_output_shapes
:*
dtype0
∞
,gcn_attention_49/core_gcn_99/conv1d_398/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,gcn_attention_49/core_gcn_99/conv1d_398/bias
©
@gcn_attention_49/core_gcn_99/conv1d_398/bias/Read/ReadVariableOpReadVariableOp,gcn_attention_49/core_gcn_99/conv1d_398/bias*
_output_shapes
:*
dtype0
ћ
:gcn_attention_49/core_gcn_99/batch_normalization_797/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_49/core_gcn_99/batch_normalization_797/gamma
≈
Ngcn_attention_49/core_gcn_99/batch_normalization_797/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_49/core_gcn_99/batch_normalization_797/gamma*
_output_shapes
:*
dtype0
 
9gcn_attention_49/core_gcn_99/batch_normalization_797/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_49/core_gcn_99/batch_normalization_797/beta
√
Mgcn_attention_49/core_gcn_99/batch_normalization_797/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_49/core_gcn_99/batch_normalization_797/beta*
_output_shapes
:*
dtype0
ћ
:gcn_attention_49/core_gcn_99/batch_normalization_798/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_49/core_gcn_99/batch_normalization_798/gamma
≈
Ngcn_attention_49/core_gcn_99/batch_normalization_798/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_49/core_gcn_99/batch_normalization_798/gamma*
_output_shapes
:*
dtype0
 
9gcn_attention_49/core_gcn_99/batch_normalization_798/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_49/core_gcn_99/batch_normalization_798/beta
√
Mgcn_attention_49/core_gcn_99/batch_normalization_798/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_49/core_gcn_99/batch_normalization_798/beta*
_output_shapes
:*
dtype0
ћ
:gcn_attention_49/core_gcn_99/batch_normalization_799/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:gcn_attention_49/core_gcn_99/batch_normalization_799/gamma
≈
Ngcn_attention_49/core_gcn_99/batch_normalization_799/gamma/Read/ReadVariableOpReadVariableOp:gcn_attention_49/core_gcn_99/batch_normalization_799/gamma*
_output_shapes
:*
dtype0
 
9gcn_attention_49/core_gcn_99/batch_normalization_799/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9gcn_attention_49/core_gcn_99/batch_normalization_799/beta
√
Mgcn_attention_49/core_gcn_99/batch_normalization_799/beta/Read/ReadVariableOpReadVariableOp9gcn_attention_49/core_gcn_99/batch_normalization_799/beta*
_output_shapes
:*
dtype0
Ў
@gcn_attention_49/core_gcn_98/batch_normalization_788/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_49/core_gcn_98/batch_normalization_788/moving_mean
—
Tgcn_attention_49/core_gcn_98/batch_normalization_788/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_49/core_gcn_98/batch_normalization_788/moving_mean*
_output_shapes
:*
dtype0
а
Dgcn_attention_49/core_gcn_98/batch_normalization_788/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_49/core_gcn_98/batch_normalization_788/moving_variance
ў
Xgcn_attention_49/core_gcn_98/batch_normalization_788/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_49/core_gcn_98/batch_normalization_788/moving_variance*
_output_shapes
:*
dtype0
Ў
@gcn_attention_49/core_gcn_98/batch_normalization_789/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_49/core_gcn_98/batch_normalization_789/moving_mean
—
Tgcn_attention_49/core_gcn_98/batch_normalization_789/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_49/core_gcn_98/batch_normalization_789/moving_mean*
_output_shapes
:*
dtype0
а
Dgcn_attention_49/core_gcn_98/batch_normalization_789/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_49/core_gcn_98/batch_normalization_789/moving_variance
ў
Xgcn_attention_49/core_gcn_98/batch_normalization_789/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_49/core_gcn_98/batch_normalization_789/moving_variance*
_output_shapes
:*
dtype0
Ў
@gcn_attention_49/core_gcn_98/batch_normalization_790/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_49/core_gcn_98/batch_normalization_790/moving_mean
—
Tgcn_attention_49/core_gcn_98/batch_normalization_790/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_49/core_gcn_98/batch_normalization_790/moving_mean*
_output_shapes
:*
dtype0
а
Dgcn_attention_49/core_gcn_98/batch_normalization_790/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_49/core_gcn_98/batch_normalization_790/moving_variance
ў
Xgcn_attention_49/core_gcn_98/batch_normalization_790/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_49/core_gcn_98/batch_normalization_790/moving_variance*
_output_shapes
:*
dtype0
Ў
@gcn_attention_49/core_gcn_98/batch_normalization_791/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_49/core_gcn_98/batch_normalization_791/moving_mean
—
Tgcn_attention_49/core_gcn_98/batch_normalization_791/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_49/core_gcn_98/batch_normalization_791/moving_mean*
_output_shapes
:*
dtype0
а
Dgcn_attention_49/core_gcn_98/batch_normalization_791/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_49/core_gcn_98/batch_normalization_791/moving_variance
ў
Xgcn_attention_49/core_gcn_98/batch_normalization_791/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_49/core_gcn_98/batch_normalization_791/moving_variance*
_output_shapes
:*
dtype0
Ў
@gcn_attention_49/core_gcn_98/batch_normalization_792/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_49/core_gcn_98/batch_normalization_792/moving_mean
—
Tgcn_attention_49/core_gcn_98/batch_normalization_792/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_49/core_gcn_98/batch_normalization_792/moving_mean*
_output_shapes
:*
dtype0
а
Dgcn_attention_49/core_gcn_98/batch_normalization_792/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_49/core_gcn_98/batch_normalization_792/moving_variance
ў
Xgcn_attention_49/core_gcn_98/batch_normalization_792/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_49/core_gcn_98/batch_normalization_792/moving_variance*
_output_shapes
:*
dtype0
Ў
@gcn_attention_49/core_gcn_98/batch_normalization_793/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_49/core_gcn_98/batch_normalization_793/moving_mean
—
Tgcn_attention_49/core_gcn_98/batch_normalization_793/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_49/core_gcn_98/batch_normalization_793/moving_mean*
_output_shapes
:*
dtype0
а
Dgcn_attention_49/core_gcn_98/batch_normalization_793/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_49/core_gcn_98/batch_normalization_793/moving_variance
ў
Xgcn_attention_49/core_gcn_98/batch_normalization_793/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_49/core_gcn_98/batch_normalization_793/moving_variance*
_output_shapes
:*
dtype0
Ў
@gcn_attention_49/core_gcn_99/batch_normalization_794/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_49/core_gcn_99/batch_normalization_794/moving_mean
—
Tgcn_attention_49/core_gcn_99/batch_normalization_794/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_49/core_gcn_99/batch_normalization_794/moving_mean*
_output_shapes
:*
dtype0
а
Dgcn_attention_49/core_gcn_99/batch_normalization_794/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_49/core_gcn_99/batch_normalization_794/moving_variance
ў
Xgcn_attention_49/core_gcn_99/batch_normalization_794/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_49/core_gcn_99/batch_normalization_794/moving_variance*
_output_shapes
:*
dtype0
Ў
@gcn_attention_49/core_gcn_99/batch_normalization_795/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_49/core_gcn_99/batch_normalization_795/moving_mean
—
Tgcn_attention_49/core_gcn_99/batch_normalization_795/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_49/core_gcn_99/batch_normalization_795/moving_mean*
_output_shapes
:*
dtype0
а
Dgcn_attention_49/core_gcn_99/batch_normalization_795/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_49/core_gcn_99/batch_normalization_795/moving_variance
ў
Xgcn_attention_49/core_gcn_99/batch_normalization_795/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_49/core_gcn_99/batch_normalization_795/moving_variance*
_output_shapes
:*
dtype0
Ў
@gcn_attention_49/core_gcn_99/batch_normalization_796/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_49/core_gcn_99/batch_normalization_796/moving_mean
—
Tgcn_attention_49/core_gcn_99/batch_normalization_796/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_49/core_gcn_99/batch_normalization_796/moving_mean*
_output_shapes
:*
dtype0
а
Dgcn_attention_49/core_gcn_99/batch_normalization_796/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_49/core_gcn_99/batch_normalization_796/moving_variance
ў
Xgcn_attention_49/core_gcn_99/batch_normalization_796/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_49/core_gcn_99/batch_normalization_796/moving_variance*
_output_shapes
:*
dtype0
Ў
@gcn_attention_49/core_gcn_99/batch_normalization_797/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_49/core_gcn_99/batch_normalization_797/moving_mean
—
Tgcn_attention_49/core_gcn_99/batch_normalization_797/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_49/core_gcn_99/batch_normalization_797/moving_mean*
_output_shapes
:*
dtype0
а
Dgcn_attention_49/core_gcn_99/batch_normalization_797/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_49/core_gcn_99/batch_normalization_797/moving_variance
ў
Xgcn_attention_49/core_gcn_99/batch_normalization_797/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_49/core_gcn_99/batch_normalization_797/moving_variance*
_output_shapes
:*
dtype0
Ў
@gcn_attention_49/core_gcn_99/batch_normalization_798/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_49/core_gcn_99/batch_normalization_798/moving_mean
—
Tgcn_attention_49/core_gcn_99/batch_normalization_798/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_49/core_gcn_99/batch_normalization_798/moving_mean*
_output_shapes
:*
dtype0
а
Dgcn_attention_49/core_gcn_99/batch_normalization_798/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_49/core_gcn_99/batch_normalization_798/moving_variance
ў
Xgcn_attention_49/core_gcn_99/batch_normalization_798/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_49/core_gcn_99/batch_normalization_798/moving_variance*
_output_shapes
:*
dtype0
Ў
@gcn_attention_49/core_gcn_99/batch_normalization_799/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@gcn_attention_49/core_gcn_99/batch_normalization_799/moving_mean
—
Tgcn_attention_49/core_gcn_99/batch_normalization_799/moving_mean/Read/ReadVariableOpReadVariableOp@gcn_attention_49/core_gcn_99/batch_normalization_799/moving_mean*
_output_shapes
:*
dtype0
а
Dgcn_attention_49/core_gcn_99/batch_normalization_799/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDgcn_attention_49/core_gcn_99/batch_normalization_799/moving_variance
ў
Xgcn_attention_49/core_gcn_99/batch_normalization_799/moving_variance/Read/ReadVariableOpReadVariableOpDgcn_attention_49/core_gcn_99/batch_normalization_799/moving_variance*
_output_shapes
:*
dtype0

NoOpNoOp
њэ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*щь
valueоьBкь Bвь
*
self_attention_layer

signatures
§
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
trainable_variables
regularization_losses
	variables
	keras_api
 
h

kernel
bias
trainable_variables
regularization_losses
 	variables
!	keras_api
R
"trainable_variables
#regularization_losses
$	variables
%	keras_api
Ч
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+trainable_variables
,regularization_losses
-	variables
.	keras_api
R
/trainable_variables
0regularization_losses
1	variables
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
8trainable_variables
9regularization_losses
:	variables
;	keras_api
h

<kernel
=bias
>trainable_variables
?regularization_losses
@	variables
A	keras_api
R
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
Ч
Faxis
	Ggamma
Hbeta
Imoving_mean
Jmoving_variance
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
R
Otrainable_variables
Pregularization_losses
Q	variables
R	keras_api
b
S
embeddings
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
R
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
Ч
\axis
	]gamma
^beta
_moving_mean
`moving_variance
atrainable_variables
bregularization_losses
c	variables
d	keras_api
R
etrainable_variables
fregularization_losses
g	variables
h	keras_api

i0
j1

k	keras_api
h

lkernel
mbias
ntrainable_variables
oregularization_losses
p	variables
q	keras_api
R
rtrainable_variables
sregularization_losses
t	variables
u	keras_api
ƒ
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
А22
Б23
В24
Г25
Д26
Е27
Ж28
З29
И30
Й31
К32
Л33
М34
Н35
О36
П37
Р38
С39
Т40
У41
Ф42
Х43
Ц44
Ч45
Ш46
Щ47
Ъ48
Ы49
Ь50
Э51
l52
m53
 
ћ
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
А22
Б23
В24
Г25
Д26
Е27
Ж28
З29
И30
Й31
К32
Л33
М34
Н35
О36
П37
Р38
С39
Т40
У41
Ф42
Х43
Ц44
Ч45
Ш46
Щ47
Ъ48
Ы49
Ь50
Э51
l52
m53
)54
*55
I56
J57
_58
`59
Ю60
Я61
†62
°63
Ґ64
£65
§66
•67
¶68
І69
®70
©71
™72
Ђ73
ђ74
≠75
Ѓ76
ѓ77
∞78
±79
≤80
≥81
і82
µ83
≤
 ґlayer_regularization_losses
trainable_variables
regularization_losses
Јmetrics
	variables
Єlayers
єnon_trainable_variables
Їlayer_metrics
{y
VARIABLE_VALUE"gcn_attention_49/conv1d_392/kernelCself_attention_layer/node0_Conv1D/kernel/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE gcn_attention_49/conv1d_392/biasAself_attention_layer/node0_Conv1D/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
≤
 їlayer_regularization_losses
trainable_variables
regularization_losses
Љmetrics
 	variables
љlayers
Њnon_trainable_variables
њlayer_metrics
 
 
 
≤
 јlayer_regularization_losses
"trainable_variables
#regularization_losses
Ѕmetrics
$	variables
¬layers
√non_trainable_variables
ƒlayer_metrics
 
УР
VARIABLE_VALUE.gcn_attention_49/batch_normalization_784/gammaNself_attention_layer/node0_BatchNormalization/gamma/.ATTRIBUTES/VARIABLE_VALUE
СО
VARIABLE_VALUE-gcn_attention_49/batch_normalization_784/betaMself_attention_layer/node0_BatchNormalization/beta/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE4gcn_attention_49/batch_normalization_784/moving_meanTself_attention_layer/node0_BatchNormalization/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
І§
VARIABLE_VALUE8gcn_attention_49/batch_normalization_784/moving_varianceXself_attention_layer/node0_BatchNormalization/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

'0
(1
 

'0
(1
)2
*3
≤
 ≈layer_regularization_losses
+trainable_variables
,regularization_losses
∆metrics
-	variables
«layers
»non_trainable_variables
…layer_metrics
 
 
 
≤
  layer_regularization_losses
/trainable_variables
0regularization_losses
Ћmetrics
1	variables
ћlayers
Ќnon_trainable_variables
ќlayer_metrics
 
 
 
 
{y
VARIABLE_VALUE gcn_attention_49/dense_49/kernelEself_attention_layer/distance_Dense/kernel/.ATTRIBUTES/VARIABLE_VALUE

70
 

70
≤
 ѕlayer_regularization_losses
8trainable_variables
9regularization_losses
–metrics
:	variables
—layers
“non_trainable_variables
”layer_metrics
ЖГ
VARIABLE_VALUE"gcn_attention_49/conv2d_197/kernelMself_attention_layer/distance_Convolution2D/kernel/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUE gcn_attention_49/conv2d_197/biasKself_attention_layer/distance_Convolution2D/bias/.ATTRIBUTES/VARIABLE_VALUE

<0
=1
 

<0
=1
≤
 ‘layer_regularization_losses
>trainable_variables
?regularization_losses
’metrics
@	variables
÷layers
„non_trainable_variables
Ўlayer_metrics
 
 
 
≤
 ўlayer_regularization_losses
Btrainable_variables
Cregularization_losses
Џmetrics
D	variables
џlayers
№non_trainable_variables
Ёlayer_metrics
 
ЦУ
VARIABLE_VALUE.gcn_attention_49/batch_normalization_786/gammaQself_attention_layer/distance_BatchNormalization/gamma/.ATTRIBUTES/VARIABLE_VALUE
ФС
VARIABLE_VALUE-gcn_attention_49/batch_normalization_786/betaPself_attention_layer/distance_BatchNormalization/beta/.ATTRIBUTES/VARIABLE_VALUE
ҐЯ
VARIABLE_VALUE4gcn_attention_49/batch_normalization_786/moving_meanWself_attention_layer/distance_BatchNormalization/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
™І
VARIABLE_VALUE8gcn_attention_49/batch_normalization_786/moving_variance[self_attention_layer/distance_BatchNormalization/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

G0
H1
 

G0
H1
I2
J3
≤
 ёlayer_regularization_losses
Ktrainable_variables
Lregularization_losses
яmetrics
M	variables
аlayers
бnon_trainable_variables
вlayer_metrics
 
 
 
≤
 гlayer_regularization_losses
Otrainable_variables
Pregularization_losses
дmetrics
Q	variables
еlayers
жnon_trainable_variables
зlayer_metrics
ГА
VARIABLE_VALUE(gcn_attention_49/embedding_49/embeddingsDself_attention_layer/adj_Dense/embeddings/.ATTRIBUTES/VARIABLE_VALUE

S0
 

S0
≤
 иlayer_regularization_losses
Ttrainable_variables
Uregularization_losses
йmetrics
V	variables
кlayers
лnon_trainable_variables
мlayer_metrics
 
 
 
≤
 нlayer_regularization_losses
Xtrainable_variables
Yregularization_losses
оmetrics
Z	variables
пlayers
рnon_trainable_variables
сlayer_metrics
 
УР
VARIABLE_VALUE.gcn_attention_49/batch_normalization_787/gammaNself_attention_layer/node_BatchNormalization5/gamma/.ATTRIBUTES/VARIABLE_VALUE
СО
VARIABLE_VALUE-gcn_attention_49/batch_normalization_787/betaMself_attention_layer/node_BatchNormalization5/beta/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE4gcn_attention_49/batch_normalization_787/moving_meanTself_attention_layer/node_BatchNormalization5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
І§
VARIABLE_VALUE8gcn_attention_49/batch_normalization_787/moving_varianceXself_attention_layer/node_BatchNormalization5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

]0
^1
 

]0
^1
_2
`3
≤
 тlayer_regularization_losses
atrainable_variables
bregularization_losses
уmetrics
c	variables
фlayers
хnon_trainable_variables
цlayer_metrics
 
 
 
≤
 чlayer_regularization_losses
etrainable_variables
fregularization_losses
шmetrics
g	variables
щlayers
ъnon_trainable_variables
ыlayer_metrics
ƒ
ьconv2d_1
эbatch_norm_1
ю	dropout_1
€conv1d_1
Аbatch_norm_2
Б	dropout_2
Вconv1d_2
Гbatch_norm_3
Д	dropout_3
Еconv1d_3
Жbatch_norm_4
З	dropout_4
Иbatch_norm_5
Йbatch_norm_6
Кtrainable_variables
Лregularization_losses
М	variables
Н	keras_api
ƒ
Оconv2d_1
Пbatch_norm_1
Р	dropout_1
Сconv1d_1
Тbatch_norm_2
У	dropout_2
Фconv1d_2
Хbatch_norm_3
Ц	dropout_3
Чconv1d_3
Шbatch_norm_4
Щ	dropout_4
Ъbatch_norm_5
Ыbatch_norm_6
Ьtrainable_variables
Эregularization_losses
Ю	variables
Я	keras_api
 
~|
VARIABLE_VALUE"gcn_attention_49/conv1d_399/kernelFself_attention_layer/node_Conv1D_out/kernel/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE gcn_attention_49/conv1d_399/biasDself_attention_layer/node_Conv1D_out/bias/.ATTRIBUTES/VARIABLE_VALUE

l0
m1
 

l0
m1
≤
 †layer_regularization_losses
ntrainable_variables
oregularization_losses
°metrics
p	variables
Ґlayers
£non_trainable_variables
§layer_metrics
 
 
 
≤
 •layer_regularization_losses
rtrainable_variables
sregularization_losses
¶metrics
t	variables
Іlayers
®non_trainable_variables
©layer_metrics
ЛИ
VARIABLE_VALUE.gcn_attention_49/core_gcn_98/conv2d_198/kernelFself_attention_layer/trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE,gcn_attention_49/core_gcn_98/conv2d_198/biasFself_attention_layer/trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE:gcn_attention_49/core_gcn_98/batch_normalization_788/gammaFself_attention_layer/trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE9gcn_attention_49/core_gcn_98/batch_normalization_788/betaFself_attention_layer/trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE.gcn_attention_49/core_gcn_98/conv1d_393/kernelFself_attention_layer/trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE,gcn_attention_49/core_gcn_98/conv1d_393/biasFself_attention_layer/trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE:gcn_attention_49/core_gcn_98/batch_normalization_789/gammaFself_attention_layer/trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE9gcn_attention_49/core_gcn_98/batch_normalization_789/betaFself_attention_layer/trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE.gcn_attention_49/core_gcn_98/conv1d_394/kernelFself_attention_layer/trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE,gcn_attention_49/core_gcn_98/conv1d_394/biasFself_attention_layer/trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE:gcn_attention_49/core_gcn_98/batch_normalization_790/gammaFself_attention_layer/trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE9gcn_attention_49/core_gcn_98/batch_normalization_790/betaFself_attention_layer/trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE.gcn_attention_49/core_gcn_98/conv1d_395/kernelFself_attention_layer/trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE,gcn_attention_49/core_gcn_98/conv1d_395/biasFself_attention_layer/trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE:gcn_attention_49/core_gcn_98/batch_normalization_791/gammaFself_attention_layer/trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE9gcn_attention_49/core_gcn_98/batch_normalization_791/betaFself_attention_layer/trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE:gcn_attention_49/core_gcn_98/batch_normalization_792/gammaFself_attention_layer/trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE9gcn_attention_49/core_gcn_98/batch_normalization_792/betaFself_attention_layer/trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE:gcn_attention_49/core_gcn_98/batch_normalization_793/gammaFself_attention_layer/trainable_variables/30/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE9gcn_attention_49/core_gcn_98/batch_normalization_793/betaFself_attention_layer/trainable_variables/31/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE.gcn_attention_49/core_gcn_99/conv2d_199/kernelFself_attention_layer/trainable_variables/32/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE,gcn_attention_49/core_gcn_99/conv2d_199/biasFself_attention_layer/trainable_variables/33/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE:gcn_attention_49/core_gcn_99/batch_normalization_794/gammaFself_attention_layer/trainable_variables/34/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE9gcn_attention_49/core_gcn_99/batch_normalization_794/betaFself_attention_layer/trainable_variables/35/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE.gcn_attention_49/core_gcn_99/conv1d_396/kernelFself_attention_layer/trainable_variables/36/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE,gcn_attention_49/core_gcn_99/conv1d_396/biasFself_attention_layer/trainable_variables/37/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE:gcn_attention_49/core_gcn_99/batch_normalization_795/gammaFself_attention_layer/trainable_variables/38/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE9gcn_attention_49/core_gcn_99/batch_normalization_795/betaFself_attention_layer/trainable_variables/39/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE.gcn_attention_49/core_gcn_99/conv1d_397/kernelFself_attention_layer/trainable_variables/40/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE,gcn_attention_49/core_gcn_99/conv1d_397/biasFself_attention_layer/trainable_variables/41/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE:gcn_attention_49/core_gcn_99/batch_normalization_796/gammaFself_attention_layer/trainable_variables/42/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE9gcn_attention_49/core_gcn_99/batch_normalization_796/betaFself_attention_layer/trainable_variables/43/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE.gcn_attention_49/core_gcn_99/conv1d_398/kernelFself_attention_layer/trainable_variables/44/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE,gcn_attention_49/core_gcn_99/conv1d_398/biasFself_attention_layer/trainable_variables/45/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE:gcn_attention_49/core_gcn_99/batch_normalization_797/gammaFself_attention_layer/trainable_variables/46/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE9gcn_attention_49/core_gcn_99/batch_normalization_797/betaFself_attention_layer/trainable_variables/47/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE:gcn_attention_49/core_gcn_99/batch_normalization_798/gammaFself_attention_layer/trainable_variables/48/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE9gcn_attention_49/core_gcn_99/batch_normalization_798/betaFself_attention_layer/trainable_variables/49/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE:gcn_attention_49/core_gcn_99/batch_normalization_799/gammaFself_attention_layer/trainable_variables/50/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE9gcn_attention_49/core_gcn_99/batch_normalization_799/betaFself_attention_layer/trainable_variables/51/.ATTRIBUTES/VARIABLE_VALUE
УР
VARIABLE_VALUE@gcn_attention_49/core_gcn_98/batch_normalization_788/moving_mean<self_attention_layer/variables/60/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUEDgcn_attention_49/core_gcn_98/batch_normalization_788/moving_variance<self_attention_layer/variables/61/.ATTRIBUTES/VARIABLE_VALUE
УР
VARIABLE_VALUE@gcn_attention_49/core_gcn_98/batch_normalization_789/moving_mean<self_attention_layer/variables/62/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUEDgcn_attention_49/core_gcn_98/batch_normalization_789/moving_variance<self_attention_layer/variables/63/.ATTRIBUTES/VARIABLE_VALUE
УР
VARIABLE_VALUE@gcn_attention_49/core_gcn_98/batch_normalization_790/moving_mean<self_attention_layer/variables/64/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUEDgcn_attention_49/core_gcn_98/batch_normalization_790/moving_variance<self_attention_layer/variables/65/.ATTRIBUTES/VARIABLE_VALUE
УР
VARIABLE_VALUE@gcn_attention_49/core_gcn_98/batch_normalization_791/moving_mean<self_attention_layer/variables/66/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUEDgcn_attention_49/core_gcn_98/batch_normalization_791/moving_variance<self_attention_layer/variables/67/.ATTRIBUTES/VARIABLE_VALUE
УР
VARIABLE_VALUE@gcn_attention_49/core_gcn_98/batch_normalization_792/moving_mean<self_attention_layer/variables/68/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUEDgcn_attention_49/core_gcn_98/batch_normalization_792/moving_variance<self_attention_layer/variables/69/.ATTRIBUTES/VARIABLE_VALUE
УР
VARIABLE_VALUE@gcn_attention_49/core_gcn_98/batch_normalization_793/moving_mean<self_attention_layer/variables/70/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUEDgcn_attention_49/core_gcn_98/batch_normalization_793/moving_variance<self_attention_layer/variables/71/.ATTRIBUTES/VARIABLE_VALUE
УР
VARIABLE_VALUE@gcn_attention_49/core_gcn_99/batch_normalization_794/moving_mean<self_attention_layer/variables/72/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUEDgcn_attention_49/core_gcn_99/batch_normalization_794/moving_variance<self_attention_layer/variables/73/.ATTRIBUTES/VARIABLE_VALUE
УР
VARIABLE_VALUE@gcn_attention_49/core_gcn_99/batch_normalization_795/moving_mean<self_attention_layer/variables/74/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUEDgcn_attention_49/core_gcn_99/batch_normalization_795/moving_variance<self_attention_layer/variables/75/.ATTRIBUTES/VARIABLE_VALUE
УР
VARIABLE_VALUE@gcn_attention_49/core_gcn_99/batch_normalization_796/moving_mean<self_attention_layer/variables/76/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUEDgcn_attention_49/core_gcn_99/batch_normalization_796/moving_variance<self_attention_layer/variables/77/.ATTRIBUTES/VARIABLE_VALUE
УР
VARIABLE_VALUE@gcn_attention_49/core_gcn_99/batch_normalization_797/moving_mean<self_attention_layer/variables/78/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUEDgcn_attention_49/core_gcn_99/batch_normalization_797/moving_variance<self_attention_layer/variables/79/.ATTRIBUTES/VARIABLE_VALUE
УР
VARIABLE_VALUE@gcn_attention_49/core_gcn_99/batch_normalization_798/moving_mean<self_attention_layer/variables/80/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUEDgcn_attention_49/core_gcn_99/batch_normalization_798/moving_variance<self_attention_layer/variables/81/.ATTRIBUTES/VARIABLE_VALUE
УР
VARIABLE_VALUE@gcn_attention_49/core_gcn_99/batch_normalization_799/moving_mean<self_attention_layer/variables/82/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUEDgcn_attention_49/core_gcn_99/batch_normalization_799/moving_variance<self_attention_layer/variables/83/.ATTRIBUTES/VARIABLE_VALUE
 
 
¶
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
ю
)0
*1
I2
J3
_4
`5
Ю6
Я7
†8
°9
Ґ10
£11
§12
•13
¶14
І15
®16
©17
™18
Ђ19
ђ20
≠21
Ѓ22
ѓ23
∞24
±25
≤26
≥27
і28
µ29
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
l

vkernel
wbias
™trainable_variables
Ђregularization_losses
ђ	variables
≠	keras_api
Ю
	Ѓaxis
	xgamma
ybeta
Юmoving_mean
Яmoving_variance
ѓtrainable_variables
∞regularization_losses
±	variables
≤	keras_api
V
≥trainable_variables
іregularization_losses
µ	variables
ґ	keras_api
l

zkernel
{bias
Јtrainable_variables
Єregularization_losses
є	variables
Ї	keras_api
Ю
	їaxis
	|gamma
}beta
†moving_mean
°moving_variance
Љtrainable_variables
љregularization_losses
Њ	variables
њ	keras_api
V
јtrainable_variables
Ѕregularization_losses
¬	variables
√	keras_api
l

~kernel
bias
ƒtrainable_variables
≈regularization_losses
∆	variables
«	keras_api
†
	»axis

Аgamma
	Бbeta
Ґmoving_mean
£moving_variance
…trainable_variables
 regularization_losses
Ћ	variables
ћ	keras_api
V
Ќtrainable_variables
ќregularization_losses
ѕ	variables
–	keras_api
n
Вkernel
	Гbias
—trainable_variables
“regularization_losses
”	variables
‘	keras_api
†
	’axis

Дgamma
	Еbeta
§moving_mean
•moving_variance
÷trainable_variables
„regularization_losses
Ў	variables
ў	keras_api
V
Џtrainable_variables
џregularization_losses
№	variables
Ё	keras_api
†
	ёaxis

Жgamma
	Зbeta
¶moving_mean
Іmoving_variance
яtrainable_variables
аregularization_losses
б	variables
в	keras_api
†
	гaxis

Иgamma
	Йbeta
®moving_mean
©moving_variance
дtrainable_variables
еregularization_losses
ж	variables
з	keras_api
†
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
А10
Б11
В12
Г13
Д14
Е15
Ж16
З17
И18
Й19
 
М
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
А10
Б11
В12
Г13
Д14
Е15
Ж16
З17
И18
Й19
Ю20
Я21
†22
°23
Ґ24
£25
§26
•27
¶28
І29
®30
©31
µ
 иlayer_regularization_losses
Кtrainable_variables
Лregularization_losses
йmetrics
М	variables
кlayers
лnon_trainable_variables
мlayer_metrics
n
Кkernel
	Лbias
нtrainable_variables
оregularization_losses
п	variables
р	keras_api
†
	сaxis

Мgamma
	Нbeta
™moving_mean
Ђmoving_variance
тtrainable_variables
уregularization_losses
ф	variables
х	keras_api
V
цtrainable_variables
чregularization_losses
ш	variables
щ	keras_api
n
Оkernel
	Пbias
ъtrainable_variables
ыregularization_losses
ь	variables
э	keras_api
†
	юaxis

Рgamma
	Сbeta
ђmoving_mean
≠moving_variance
€trainable_variables
Аregularization_losses
Б	variables
В	keras_api
V
Гtrainable_variables
Дregularization_losses
Е	variables
Ж	keras_api
n
Тkernel
	Уbias
Зtrainable_variables
Иregularization_losses
Й	variables
К	keras_api
†
	Лaxis

Фgamma
	Хbeta
Ѓmoving_mean
ѓmoving_variance
Мtrainable_variables
Нregularization_losses
О	variables
П	keras_api
V
Рtrainable_variables
Сregularization_losses
Т	variables
У	keras_api
n
Цkernel
	Чbias
Фtrainable_variables
Хregularization_losses
Ц	variables
Ч	keras_api
†
	Шaxis

Шgamma
	Щbeta
∞moving_mean
±moving_variance
Щtrainable_variables
Ъregularization_losses
Ы	variables
Ь	keras_api
V
Эtrainable_variables
Юregularization_losses
Я	variables
†	keras_api
†
	°axis

Ъgamma
	Ыbeta
≤moving_mean
≥moving_variance
Ґtrainable_variables
£regularization_losses
§	variables
•	keras_api
†
	¶axis

Ьgamma
	Эbeta
іmoving_mean
µmoving_variance
Іtrainable_variables
®regularization_losses
©	variables
™	keras_api
™
К0
Л1
М2
Н3
О4
П5
Р6
С7
Т8
У9
Ф10
Х11
Ц12
Ч13
Ш14
Щ15
Ъ16
Ы17
Ь18
Э19
 
Ц
К0
Л1
М2
Н3
О4
П5
Р6
С7
Т8
У9
Ф10
Х11
Ц12
Ч13
Ш14
Щ15
Ъ16
Ы17
Ь18
Э19
™20
Ђ21
ђ22
≠23
Ѓ24
ѓ25
∞26
±27
≤28
≥29
і30
µ31
µ
 Ђlayer_regularization_losses
Ьtrainable_variables
Эregularization_losses
ђmetrics
Ю	variables
≠layers
Ѓnon_trainable_variables
ѓlayer_metrics
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
 

v0
w1
µ
 ∞layer_regularization_losses
™trainable_variables
Ђregularization_losses
±metrics
ђ	variables
≤layers
≥non_trainable_variables
іlayer_metrics
 

x0
y1
 

x0
y1
Ю2
Я3
µ
 µlayer_regularization_losses
ѓtrainable_variables
∞regularization_losses
ґmetrics
±	variables
Јlayers
Єnon_trainable_variables
єlayer_metrics
 
 
 
µ
 Їlayer_regularization_losses
≥trainable_variables
іregularization_losses
їmetrics
µ	variables
Љlayers
љnon_trainable_variables
Њlayer_metrics

z0
{1
 

z0
{1
µ
 њlayer_regularization_losses
Јtrainable_variables
Єregularization_losses
јmetrics
є	variables
Ѕlayers
¬non_trainable_variables
√layer_metrics
 

|0
}1
 

|0
}1
†2
°3
µ
 ƒlayer_regularization_losses
Љtrainable_variables
љregularization_losses
≈metrics
Њ	variables
∆layers
«non_trainable_variables
»layer_metrics
 
 
 
µ
 …layer_regularization_losses
јtrainable_variables
Ѕregularization_losses
 metrics
¬	variables
Ћlayers
ћnon_trainable_variables
Ќlayer_metrics

~0
1
 

~0
1
µ
 ќlayer_regularization_losses
ƒtrainable_variables
≈regularization_losses
ѕmetrics
∆	variables
–layers
—non_trainable_variables
“layer_metrics
 

А0
Б1
 
 
А0
Б1
Ґ2
£3
µ
 ”layer_regularization_losses
…trainable_variables
 regularization_losses
‘metrics
Ћ	variables
’layers
÷non_trainable_variables
„layer_metrics
 
 
 
µ
 Ўlayer_regularization_losses
Ќtrainable_variables
ќregularization_losses
ўmetrics
ѕ	variables
Џlayers
џnon_trainable_variables
№layer_metrics

В0
Г1
 

В0
Г1
µ
 Ёlayer_regularization_losses
—trainable_variables
“regularization_losses
ёmetrics
”	variables
яlayers
аnon_trainable_variables
бlayer_metrics
 

Д0
Е1
 
 
Д0
Е1
§2
•3
µ
 вlayer_regularization_losses
÷trainable_variables
„regularization_losses
гmetrics
Ў	variables
дlayers
еnon_trainable_variables
жlayer_metrics
 
 
 
µ
 зlayer_regularization_losses
Џtrainable_variables
џregularization_losses
иmetrics
№	variables
йlayers
кnon_trainable_variables
лlayer_metrics
 

Ж0
З1
 
 
Ж0
З1
¶2
І3
µ
 мlayer_regularization_losses
яtrainable_variables
аregularization_losses
нmetrics
б	variables
оlayers
пnon_trainable_variables
рlayer_metrics
 

И0
Й1
 
 
И0
Й1
®2
©3
µ
 сlayer_regularization_losses
дtrainable_variables
еregularization_losses
тmetrics
ж	variables
уlayers
фnon_trainable_variables
хlayer_metrics
 
 
t
ь0
э1
ю2
€3
А4
Б5
В6
Г7
Д8
Е9
Ж10
З11
И12
Й13
b
Ю0
Я1
†2
°3
Ґ4
£5
§6
•7
¶8
І9
®10
©11
 

К0
Л1
 

К0
Л1
µ
 цlayer_regularization_losses
нtrainable_variables
оregularization_losses
чmetrics
п	variables
шlayers
щnon_trainable_variables
ъlayer_metrics
 

М0
Н1
 
 
М0
Н1
™2
Ђ3
µ
 ыlayer_regularization_losses
тtrainable_variables
уregularization_losses
ьmetrics
ф	variables
эlayers
юnon_trainable_variables
€layer_metrics
 
 
 
µ
 Аlayer_regularization_losses
цtrainable_variables
чregularization_losses
Бmetrics
ш	variables
Вlayers
Гnon_trainable_variables
Дlayer_metrics

О0
П1
 

О0
П1
µ
 Еlayer_regularization_losses
ъtrainable_variables
ыregularization_losses
Жmetrics
ь	variables
Зlayers
Иnon_trainable_variables
Йlayer_metrics
 

Р0
С1
 
 
Р0
С1
ђ2
≠3
µ
 Кlayer_regularization_losses
€trainable_variables
Аregularization_losses
Лmetrics
Б	variables
Мlayers
Нnon_trainable_variables
Оlayer_metrics
 
 
 
µ
 Пlayer_regularization_losses
Гtrainable_variables
Дregularization_losses
Рmetrics
Е	variables
Сlayers
Тnon_trainable_variables
Уlayer_metrics

Т0
У1
 

Т0
У1
µ
 Фlayer_regularization_losses
Зtrainable_variables
Иregularization_losses
Хmetrics
Й	variables
Цlayers
Чnon_trainable_variables
Шlayer_metrics
 

Ф0
Х1
 
 
Ф0
Х1
Ѓ2
ѓ3
µ
 Щlayer_regularization_losses
Мtrainable_variables
Нregularization_losses
Ъmetrics
О	variables
Ыlayers
Ьnon_trainable_variables
Эlayer_metrics
 
 
 
µ
 Юlayer_regularization_losses
Рtrainable_variables
Сregularization_losses
Яmetrics
Т	variables
†layers
°non_trainable_variables
Ґlayer_metrics

Ц0
Ч1
 

Ц0
Ч1
µ
 £layer_regularization_losses
Фtrainable_variables
Хregularization_losses
§metrics
Ц	variables
•layers
¶non_trainable_variables
Іlayer_metrics
 

Ш0
Щ1
 
 
Ш0
Щ1
∞2
±3
µ
 ®layer_regularization_losses
Щtrainable_variables
Ъregularization_losses
©metrics
Ы	variables
™layers
Ђnon_trainable_variables
ђlayer_metrics
 
 
 
µ
 ≠layer_regularization_losses
Эtrainable_variables
Юregularization_losses
Ѓmetrics
Я	variables
ѓlayers
∞non_trainable_variables
±layer_metrics
 

Ъ0
Ы1
 
 
Ъ0
Ы1
≤2
≥3
µ
 ≤layer_regularization_losses
Ґtrainable_variables
£regularization_losses
≥metrics
§	variables
іlayers
µnon_trainable_variables
ґlayer_metrics
 

Ь0
Э1
 
 
Ь0
Э1
і2
µ3
µ
 Јlayer_regularization_losses
Іtrainable_variables
®regularization_losses
Єmetrics
©	variables
єlayers
Їnon_trainable_variables
їlayer_metrics
 
 
t
О0
П1
Р2
С3
Т4
У5
Ф6
Х7
Ц8
Ч9
Ш10
Щ11
Ъ12
Ы13
b
™0
Ђ1
ђ2
≠3
Ѓ4
ѓ5
∞6
±7
≤8
≥9
і10
µ11
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
Ю0
Я1
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
†0
°1
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
Ґ0
£1
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
§0
•1
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
¶0
І1
 
 
 
 

®0
©1
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
™0
Ђ1
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
ђ0
≠1
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
Ѓ0
ѓ1
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
∞0
±1
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
≤0
≥1
 
 
 
 

і0
µ1
 
Ъ
serving_default_input_text1Placeholder*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ*
dtype0**
shape!:€€€€€€€€€€€€€€€€€€ђ
≤
serving_default_input_text2Placeholder*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
dtype0*6
shape-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
≤
serving_default_input_text3Placeholder*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
dtype0*6
shape-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Щ)
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_text1serving_default_input_text2serving_default_input_text3"gcn_attention_49/conv1d_392/kernel gcn_attention_49/conv1d_392/bias4gcn_attention_49/batch_normalization_784/moving_mean8gcn_attention_49/batch_normalization_784/moving_variance-gcn_attention_49/batch_normalization_784/beta.gcn_attention_49/batch_normalization_784/gamma gcn_attention_49/dense_49/kernel"gcn_attention_49/conv2d_197/kernel gcn_attention_49/conv2d_197/bias.gcn_attention_49/batch_normalization_786/gamma-gcn_attention_49/batch_normalization_786/beta4gcn_attention_49/batch_normalization_786/moving_mean8gcn_attention_49/batch_normalization_786/moving_variance(gcn_attention_49/embedding_49/embeddings.gcn_attention_49/core_gcn_98/conv2d_198/kernel,gcn_attention_49/core_gcn_98/conv2d_198/bias:gcn_attention_49/core_gcn_98/batch_normalization_788/gamma9gcn_attention_49/core_gcn_98/batch_normalization_788/beta@gcn_attention_49/core_gcn_98/batch_normalization_788/moving_meanDgcn_attention_49/core_gcn_98/batch_normalization_788/moving_variance.gcn_attention_49/core_gcn_98/conv1d_393/kernel,gcn_attention_49/core_gcn_98/conv1d_393/bias@gcn_attention_49/core_gcn_98/batch_normalization_789/moving_meanDgcn_attention_49/core_gcn_98/batch_normalization_789/moving_variance9gcn_attention_49/core_gcn_98/batch_normalization_789/beta:gcn_attention_49/core_gcn_98/batch_normalization_789/gamma.gcn_attention_49/core_gcn_98/conv1d_394/kernel,gcn_attention_49/core_gcn_98/conv1d_394/bias@gcn_attention_49/core_gcn_98/batch_normalization_790/moving_meanDgcn_attention_49/core_gcn_98/batch_normalization_790/moving_variance9gcn_attention_49/core_gcn_98/batch_normalization_790/beta:gcn_attention_49/core_gcn_98/batch_normalization_790/gamma.gcn_attention_49/core_gcn_98/conv1d_395/kernel,gcn_attention_49/core_gcn_98/conv1d_395/bias@gcn_attention_49/core_gcn_98/batch_normalization_791/moving_meanDgcn_attention_49/core_gcn_98/batch_normalization_791/moving_variance9gcn_attention_49/core_gcn_98/batch_normalization_791/beta:gcn_attention_49/core_gcn_98/batch_normalization_791/gamma:gcn_attention_49/core_gcn_98/batch_normalization_792/gamma9gcn_attention_49/core_gcn_98/batch_normalization_792/beta@gcn_attention_49/core_gcn_98/batch_normalization_792/moving_meanDgcn_attention_49/core_gcn_98/batch_normalization_792/moving_variance@gcn_attention_49/core_gcn_98/batch_normalization_793/moving_meanDgcn_attention_49/core_gcn_98/batch_normalization_793/moving_variance9gcn_attention_49/core_gcn_98/batch_normalization_793/beta:gcn_attention_49/core_gcn_98/batch_normalization_793/gamma.gcn_attention_49/core_gcn_99/conv2d_199/kernel,gcn_attention_49/core_gcn_99/conv2d_199/bias:gcn_attention_49/core_gcn_99/batch_normalization_794/gamma9gcn_attention_49/core_gcn_99/batch_normalization_794/beta@gcn_attention_49/core_gcn_99/batch_normalization_794/moving_meanDgcn_attention_49/core_gcn_99/batch_normalization_794/moving_variance.gcn_attention_49/core_gcn_99/conv1d_396/kernel,gcn_attention_49/core_gcn_99/conv1d_396/bias@gcn_attention_49/core_gcn_99/batch_normalization_795/moving_meanDgcn_attention_49/core_gcn_99/batch_normalization_795/moving_variance9gcn_attention_49/core_gcn_99/batch_normalization_795/beta:gcn_attention_49/core_gcn_99/batch_normalization_795/gamma.gcn_attention_49/core_gcn_99/conv1d_397/kernel,gcn_attention_49/core_gcn_99/conv1d_397/bias@gcn_attention_49/core_gcn_99/batch_normalization_796/moving_meanDgcn_attention_49/core_gcn_99/batch_normalization_796/moving_variance9gcn_attention_49/core_gcn_99/batch_normalization_796/beta:gcn_attention_49/core_gcn_99/batch_normalization_796/gamma.gcn_attention_49/core_gcn_99/conv1d_398/kernel,gcn_attention_49/core_gcn_99/conv1d_398/bias@gcn_attention_49/core_gcn_99/batch_normalization_797/moving_meanDgcn_attention_49/core_gcn_99/batch_normalization_797/moving_variance9gcn_attention_49/core_gcn_99/batch_normalization_797/beta:gcn_attention_49/core_gcn_99/batch_normalization_797/gamma:gcn_attention_49/core_gcn_99/batch_normalization_798/gamma9gcn_attention_49/core_gcn_99/batch_normalization_798/beta@gcn_attention_49/core_gcn_99/batch_normalization_798/moving_meanDgcn_attention_49/core_gcn_99/batch_normalization_798/moving_variance@gcn_attention_49/core_gcn_99/batch_normalization_799/moving_meanDgcn_attention_49/core_gcn_99/batch_normalization_799/moving_variance9gcn_attention_49/core_gcn_99/batch_normalization_799/beta:gcn_attention_49/core_gcn_99/batch_normalization_799/gamma4gcn_attention_49/batch_normalization_787/moving_mean8gcn_attention_49/batch_normalization_787/moving_variance-gcn_attention_49/batch_normalization_787/beta.gcn_attention_49/batch_normalization_787/gamma"gcn_attention_49/conv1d_399/kernel gcn_attention_49/conv1d_399/bias*b
Tin[
Y2W*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*v
_read_only_resource_inputsX
VT	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUV*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_611223
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
с4
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename6gcn_attention_49/conv1d_392/kernel/Read/ReadVariableOp4gcn_attention_49/conv1d_392/bias/Read/ReadVariableOpBgcn_attention_49/batch_normalization_784/gamma/Read/ReadVariableOpAgcn_attention_49/batch_normalization_784/beta/Read/ReadVariableOpHgcn_attention_49/batch_normalization_784/moving_mean/Read/ReadVariableOpLgcn_attention_49/batch_normalization_784/moving_variance/Read/ReadVariableOp4gcn_attention_49/dense_49/kernel/Read/ReadVariableOp6gcn_attention_49/conv2d_197/kernel/Read/ReadVariableOp4gcn_attention_49/conv2d_197/bias/Read/ReadVariableOpBgcn_attention_49/batch_normalization_786/gamma/Read/ReadVariableOpAgcn_attention_49/batch_normalization_786/beta/Read/ReadVariableOpHgcn_attention_49/batch_normalization_786/moving_mean/Read/ReadVariableOpLgcn_attention_49/batch_normalization_786/moving_variance/Read/ReadVariableOp<gcn_attention_49/embedding_49/embeddings/Read/ReadVariableOpBgcn_attention_49/batch_normalization_787/gamma/Read/ReadVariableOpAgcn_attention_49/batch_normalization_787/beta/Read/ReadVariableOpHgcn_attention_49/batch_normalization_787/moving_mean/Read/ReadVariableOpLgcn_attention_49/batch_normalization_787/moving_variance/Read/ReadVariableOp6gcn_attention_49/conv1d_399/kernel/Read/ReadVariableOp4gcn_attention_49/conv1d_399/bias/Read/ReadVariableOpBgcn_attention_49/core_gcn_98/conv2d_198/kernel/Read/ReadVariableOp@gcn_attention_49/core_gcn_98/conv2d_198/bias/Read/ReadVariableOpNgcn_attention_49/core_gcn_98/batch_normalization_788/gamma/Read/ReadVariableOpMgcn_attention_49/core_gcn_98/batch_normalization_788/beta/Read/ReadVariableOpBgcn_attention_49/core_gcn_98/conv1d_393/kernel/Read/ReadVariableOp@gcn_attention_49/core_gcn_98/conv1d_393/bias/Read/ReadVariableOpNgcn_attention_49/core_gcn_98/batch_normalization_789/gamma/Read/ReadVariableOpMgcn_attention_49/core_gcn_98/batch_normalization_789/beta/Read/ReadVariableOpBgcn_attention_49/core_gcn_98/conv1d_394/kernel/Read/ReadVariableOp@gcn_attention_49/core_gcn_98/conv1d_394/bias/Read/ReadVariableOpNgcn_attention_49/core_gcn_98/batch_normalization_790/gamma/Read/ReadVariableOpMgcn_attention_49/core_gcn_98/batch_normalization_790/beta/Read/ReadVariableOpBgcn_attention_49/core_gcn_98/conv1d_395/kernel/Read/ReadVariableOp@gcn_attention_49/core_gcn_98/conv1d_395/bias/Read/ReadVariableOpNgcn_attention_49/core_gcn_98/batch_normalization_791/gamma/Read/ReadVariableOpMgcn_attention_49/core_gcn_98/batch_normalization_791/beta/Read/ReadVariableOpNgcn_attention_49/core_gcn_98/batch_normalization_792/gamma/Read/ReadVariableOpMgcn_attention_49/core_gcn_98/batch_normalization_792/beta/Read/ReadVariableOpNgcn_attention_49/core_gcn_98/batch_normalization_793/gamma/Read/ReadVariableOpMgcn_attention_49/core_gcn_98/batch_normalization_793/beta/Read/ReadVariableOpBgcn_attention_49/core_gcn_99/conv2d_199/kernel/Read/ReadVariableOp@gcn_attention_49/core_gcn_99/conv2d_199/bias/Read/ReadVariableOpNgcn_attention_49/core_gcn_99/batch_normalization_794/gamma/Read/ReadVariableOpMgcn_attention_49/core_gcn_99/batch_normalization_794/beta/Read/ReadVariableOpBgcn_attention_49/core_gcn_99/conv1d_396/kernel/Read/ReadVariableOp@gcn_attention_49/core_gcn_99/conv1d_396/bias/Read/ReadVariableOpNgcn_attention_49/core_gcn_99/batch_normalization_795/gamma/Read/ReadVariableOpMgcn_attention_49/core_gcn_99/batch_normalization_795/beta/Read/ReadVariableOpBgcn_attention_49/core_gcn_99/conv1d_397/kernel/Read/ReadVariableOp@gcn_attention_49/core_gcn_99/conv1d_397/bias/Read/ReadVariableOpNgcn_attention_49/core_gcn_99/batch_normalization_796/gamma/Read/ReadVariableOpMgcn_attention_49/core_gcn_99/batch_normalization_796/beta/Read/ReadVariableOpBgcn_attention_49/core_gcn_99/conv1d_398/kernel/Read/ReadVariableOp@gcn_attention_49/core_gcn_99/conv1d_398/bias/Read/ReadVariableOpNgcn_attention_49/core_gcn_99/batch_normalization_797/gamma/Read/ReadVariableOpMgcn_attention_49/core_gcn_99/batch_normalization_797/beta/Read/ReadVariableOpNgcn_attention_49/core_gcn_99/batch_normalization_798/gamma/Read/ReadVariableOpMgcn_attention_49/core_gcn_99/batch_normalization_798/beta/Read/ReadVariableOpNgcn_attention_49/core_gcn_99/batch_normalization_799/gamma/Read/ReadVariableOpMgcn_attention_49/core_gcn_99/batch_normalization_799/beta/Read/ReadVariableOpTgcn_attention_49/core_gcn_98/batch_normalization_788/moving_mean/Read/ReadVariableOpXgcn_attention_49/core_gcn_98/batch_normalization_788/moving_variance/Read/ReadVariableOpTgcn_attention_49/core_gcn_98/batch_normalization_789/moving_mean/Read/ReadVariableOpXgcn_attention_49/core_gcn_98/batch_normalization_789/moving_variance/Read/ReadVariableOpTgcn_attention_49/core_gcn_98/batch_normalization_790/moving_mean/Read/ReadVariableOpXgcn_attention_49/core_gcn_98/batch_normalization_790/moving_variance/Read/ReadVariableOpTgcn_attention_49/core_gcn_98/batch_normalization_791/moving_mean/Read/ReadVariableOpXgcn_attention_49/core_gcn_98/batch_normalization_791/moving_variance/Read/ReadVariableOpTgcn_attention_49/core_gcn_98/batch_normalization_792/moving_mean/Read/ReadVariableOpXgcn_attention_49/core_gcn_98/batch_normalization_792/moving_variance/Read/ReadVariableOpTgcn_attention_49/core_gcn_98/batch_normalization_793/moving_mean/Read/ReadVariableOpXgcn_attention_49/core_gcn_98/batch_normalization_793/moving_variance/Read/ReadVariableOpTgcn_attention_49/core_gcn_99/batch_normalization_794/moving_mean/Read/ReadVariableOpXgcn_attention_49/core_gcn_99/batch_normalization_794/moving_variance/Read/ReadVariableOpTgcn_attention_49/core_gcn_99/batch_normalization_795/moving_mean/Read/ReadVariableOpXgcn_attention_49/core_gcn_99/batch_normalization_795/moving_variance/Read/ReadVariableOpTgcn_attention_49/core_gcn_99/batch_normalization_796/moving_mean/Read/ReadVariableOpXgcn_attention_49/core_gcn_99/batch_normalization_796/moving_variance/Read/ReadVariableOpTgcn_attention_49/core_gcn_99/batch_normalization_797/moving_mean/Read/ReadVariableOpXgcn_attention_49/core_gcn_99/batch_normalization_797/moving_variance/Read/ReadVariableOpTgcn_attention_49/core_gcn_99/batch_normalization_798/moving_mean/Read/ReadVariableOpXgcn_attention_49/core_gcn_99/batch_normalization_798/moving_variance/Read/ReadVariableOpTgcn_attention_49/core_gcn_99/batch_normalization_799/moving_mean/Read/ReadVariableOpXgcn_attention_49/core_gcn_99/batch_normalization_799/moving_variance/Read/ReadVariableOpConst*a
TinZ
X2V*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__traced_save_614964
№'
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename"gcn_attention_49/conv1d_392/kernel gcn_attention_49/conv1d_392/bias.gcn_attention_49/batch_normalization_784/gamma-gcn_attention_49/batch_normalization_784/beta4gcn_attention_49/batch_normalization_784/moving_mean8gcn_attention_49/batch_normalization_784/moving_variance gcn_attention_49/dense_49/kernel"gcn_attention_49/conv2d_197/kernel gcn_attention_49/conv2d_197/bias.gcn_attention_49/batch_normalization_786/gamma-gcn_attention_49/batch_normalization_786/beta4gcn_attention_49/batch_normalization_786/moving_mean8gcn_attention_49/batch_normalization_786/moving_variance(gcn_attention_49/embedding_49/embeddings.gcn_attention_49/batch_normalization_787/gamma-gcn_attention_49/batch_normalization_787/beta4gcn_attention_49/batch_normalization_787/moving_mean8gcn_attention_49/batch_normalization_787/moving_variance"gcn_attention_49/conv1d_399/kernel gcn_attention_49/conv1d_399/bias.gcn_attention_49/core_gcn_98/conv2d_198/kernel,gcn_attention_49/core_gcn_98/conv2d_198/bias:gcn_attention_49/core_gcn_98/batch_normalization_788/gamma9gcn_attention_49/core_gcn_98/batch_normalization_788/beta.gcn_attention_49/core_gcn_98/conv1d_393/kernel,gcn_attention_49/core_gcn_98/conv1d_393/bias:gcn_attention_49/core_gcn_98/batch_normalization_789/gamma9gcn_attention_49/core_gcn_98/batch_normalization_789/beta.gcn_attention_49/core_gcn_98/conv1d_394/kernel,gcn_attention_49/core_gcn_98/conv1d_394/bias:gcn_attention_49/core_gcn_98/batch_normalization_790/gamma9gcn_attention_49/core_gcn_98/batch_normalization_790/beta.gcn_attention_49/core_gcn_98/conv1d_395/kernel,gcn_attention_49/core_gcn_98/conv1d_395/bias:gcn_attention_49/core_gcn_98/batch_normalization_791/gamma9gcn_attention_49/core_gcn_98/batch_normalization_791/beta:gcn_attention_49/core_gcn_98/batch_normalization_792/gamma9gcn_attention_49/core_gcn_98/batch_normalization_792/beta:gcn_attention_49/core_gcn_98/batch_normalization_793/gamma9gcn_attention_49/core_gcn_98/batch_normalization_793/beta.gcn_attention_49/core_gcn_99/conv2d_199/kernel,gcn_attention_49/core_gcn_99/conv2d_199/bias:gcn_attention_49/core_gcn_99/batch_normalization_794/gamma9gcn_attention_49/core_gcn_99/batch_normalization_794/beta.gcn_attention_49/core_gcn_99/conv1d_396/kernel,gcn_attention_49/core_gcn_99/conv1d_396/bias:gcn_attention_49/core_gcn_99/batch_normalization_795/gamma9gcn_attention_49/core_gcn_99/batch_normalization_795/beta.gcn_attention_49/core_gcn_99/conv1d_397/kernel,gcn_attention_49/core_gcn_99/conv1d_397/bias:gcn_attention_49/core_gcn_99/batch_normalization_796/gamma9gcn_attention_49/core_gcn_99/batch_normalization_796/beta.gcn_attention_49/core_gcn_99/conv1d_398/kernel,gcn_attention_49/core_gcn_99/conv1d_398/bias:gcn_attention_49/core_gcn_99/batch_normalization_797/gamma9gcn_attention_49/core_gcn_99/batch_normalization_797/beta:gcn_attention_49/core_gcn_99/batch_normalization_798/gamma9gcn_attention_49/core_gcn_99/batch_normalization_798/beta:gcn_attention_49/core_gcn_99/batch_normalization_799/gamma9gcn_attention_49/core_gcn_99/batch_normalization_799/beta@gcn_attention_49/core_gcn_98/batch_normalization_788/moving_meanDgcn_attention_49/core_gcn_98/batch_normalization_788/moving_variance@gcn_attention_49/core_gcn_98/batch_normalization_789/moving_meanDgcn_attention_49/core_gcn_98/batch_normalization_789/moving_variance@gcn_attention_49/core_gcn_98/batch_normalization_790/moving_meanDgcn_attention_49/core_gcn_98/batch_normalization_790/moving_variance@gcn_attention_49/core_gcn_98/batch_normalization_791/moving_meanDgcn_attention_49/core_gcn_98/batch_normalization_791/moving_variance@gcn_attention_49/core_gcn_98/batch_normalization_792/moving_meanDgcn_attention_49/core_gcn_98/batch_normalization_792/moving_variance@gcn_attention_49/core_gcn_98/batch_normalization_793/moving_meanDgcn_attention_49/core_gcn_98/batch_normalization_793/moving_variance@gcn_attention_49/core_gcn_99/batch_normalization_794/moving_meanDgcn_attention_49/core_gcn_99/batch_normalization_794/moving_variance@gcn_attention_49/core_gcn_99/batch_normalization_795/moving_meanDgcn_attention_49/core_gcn_99/batch_normalization_795/moving_variance@gcn_attention_49/core_gcn_99/batch_normalization_796/moving_meanDgcn_attention_49/core_gcn_99/batch_normalization_796/moving_variance@gcn_attention_49/core_gcn_99/batch_normalization_797/moving_meanDgcn_attention_49/core_gcn_99/batch_normalization_797/moving_variance@gcn_attention_49/core_gcn_99/batch_normalization_798/moving_meanDgcn_attention_49/core_gcn_99/batch_normalization_798/moving_variance@gcn_attention_49/core_gcn_99/batch_normalization_799/moving_meanDgcn_attention_49/core_gcn_99/batch_normalization_799/moving_variance*`
TinY
W2U*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__traced_restore_615226А≈"
ЬХ
нJ
"__inference__traced_restore_615226
file_prefixJ
3assignvariableop_gcn_attention_49_conv1d_392_kernel:ђA
3assignvariableop_1_gcn_attention_49_conv1d_392_bias:O
Aassignvariableop_2_gcn_attention_49_batch_normalization_784_gamma:N
@assignvariableop_3_gcn_attention_49_batch_normalization_784_beta:U
Gassignvariableop_4_gcn_attention_49_batch_normalization_784_moving_mean:Y
Kassignvariableop_5_gcn_attention_49_batch_normalization_784_moving_variance:E
3assignvariableop_6_gcn_attention_49_dense_49_kernel:	O
5assignvariableop_7_gcn_attention_49_conv2d_197_kernel:		A
3assignvariableop_8_gcn_attention_49_conv2d_197_bias:	O
Aassignvariableop_9_gcn_attention_49_batch_normalization_786_gamma:	O
Aassignvariableop_10_gcn_attention_49_batch_normalization_786_beta:	V
Hassignvariableop_11_gcn_attention_49_batch_normalization_786_moving_mean:	Z
Lassignvariableop_12_gcn_attention_49_batch_normalization_786_moving_variance:	N
<assignvariableop_13_gcn_attention_49_embedding_49_embeddings:	P
Bassignvariableop_14_gcn_attention_49_batch_normalization_787_gamma:O
Aassignvariableop_15_gcn_attention_49_batch_normalization_787_beta:V
Hassignvariableop_16_gcn_attention_49_batch_normalization_787_moving_mean:Z
Lassignvariableop_17_gcn_attention_49_batch_normalization_787_moving_variance:L
6assignvariableop_18_gcn_attention_49_conv1d_399_kernel:B
4assignvariableop_19_gcn_attention_49_conv1d_399_bias:\
Bassignvariableop_20_gcn_attention_49_core_gcn_98_conv2d_198_kernel:N
@assignvariableop_21_gcn_attention_49_core_gcn_98_conv2d_198_bias:\
Nassignvariableop_22_gcn_attention_49_core_gcn_98_batch_normalization_788_gamma:[
Massignvariableop_23_gcn_attention_49_core_gcn_98_batch_normalization_788_beta:X
Bassignvariableop_24_gcn_attention_49_core_gcn_98_conv1d_393_kernel:N
@assignvariableop_25_gcn_attention_49_core_gcn_98_conv1d_393_bias:\
Nassignvariableop_26_gcn_attention_49_core_gcn_98_batch_normalization_789_gamma:[
Massignvariableop_27_gcn_attention_49_core_gcn_98_batch_normalization_789_beta:X
Bassignvariableop_28_gcn_attention_49_core_gcn_98_conv1d_394_kernel:N
@assignvariableop_29_gcn_attention_49_core_gcn_98_conv1d_394_bias:\
Nassignvariableop_30_gcn_attention_49_core_gcn_98_batch_normalization_790_gamma:[
Massignvariableop_31_gcn_attention_49_core_gcn_98_batch_normalization_790_beta:X
Bassignvariableop_32_gcn_attention_49_core_gcn_98_conv1d_395_kernel:N
@assignvariableop_33_gcn_attention_49_core_gcn_98_conv1d_395_bias:\
Nassignvariableop_34_gcn_attention_49_core_gcn_98_batch_normalization_791_gamma:[
Massignvariableop_35_gcn_attention_49_core_gcn_98_batch_normalization_791_beta:\
Nassignvariableop_36_gcn_attention_49_core_gcn_98_batch_normalization_792_gamma:[
Massignvariableop_37_gcn_attention_49_core_gcn_98_batch_normalization_792_beta:\
Nassignvariableop_38_gcn_attention_49_core_gcn_98_batch_normalization_793_gamma:[
Massignvariableop_39_gcn_attention_49_core_gcn_98_batch_normalization_793_beta:\
Bassignvariableop_40_gcn_attention_49_core_gcn_99_conv2d_199_kernel:N
@assignvariableop_41_gcn_attention_49_core_gcn_99_conv2d_199_bias:\
Nassignvariableop_42_gcn_attention_49_core_gcn_99_batch_normalization_794_gamma:[
Massignvariableop_43_gcn_attention_49_core_gcn_99_batch_normalization_794_beta:X
Bassignvariableop_44_gcn_attention_49_core_gcn_99_conv1d_396_kernel:N
@assignvariableop_45_gcn_attention_49_core_gcn_99_conv1d_396_bias:\
Nassignvariableop_46_gcn_attention_49_core_gcn_99_batch_normalization_795_gamma:[
Massignvariableop_47_gcn_attention_49_core_gcn_99_batch_normalization_795_beta:X
Bassignvariableop_48_gcn_attention_49_core_gcn_99_conv1d_397_kernel:N
@assignvariableop_49_gcn_attention_49_core_gcn_99_conv1d_397_bias:\
Nassignvariableop_50_gcn_attention_49_core_gcn_99_batch_normalization_796_gamma:[
Massignvariableop_51_gcn_attention_49_core_gcn_99_batch_normalization_796_beta:X
Bassignvariableop_52_gcn_attention_49_core_gcn_99_conv1d_398_kernel:N
@assignvariableop_53_gcn_attention_49_core_gcn_99_conv1d_398_bias:\
Nassignvariableop_54_gcn_attention_49_core_gcn_99_batch_normalization_797_gamma:[
Massignvariableop_55_gcn_attention_49_core_gcn_99_batch_normalization_797_beta:\
Nassignvariableop_56_gcn_attention_49_core_gcn_99_batch_normalization_798_gamma:[
Massignvariableop_57_gcn_attention_49_core_gcn_99_batch_normalization_798_beta:\
Nassignvariableop_58_gcn_attention_49_core_gcn_99_batch_normalization_799_gamma:[
Massignvariableop_59_gcn_attention_49_core_gcn_99_batch_normalization_799_beta:b
Tassignvariableop_60_gcn_attention_49_core_gcn_98_batch_normalization_788_moving_mean:f
Xassignvariableop_61_gcn_attention_49_core_gcn_98_batch_normalization_788_moving_variance:b
Tassignvariableop_62_gcn_attention_49_core_gcn_98_batch_normalization_789_moving_mean:f
Xassignvariableop_63_gcn_attention_49_core_gcn_98_batch_normalization_789_moving_variance:b
Tassignvariableop_64_gcn_attention_49_core_gcn_98_batch_normalization_790_moving_mean:f
Xassignvariableop_65_gcn_attention_49_core_gcn_98_batch_normalization_790_moving_variance:b
Tassignvariableop_66_gcn_attention_49_core_gcn_98_batch_normalization_791_moving_mean:f
Xassignvariableop_67_gcn_attention_49_core_gcn_98_batch_normalization_791_moving_variance:b
Tassignvariableop_68_gcn_attention_49_core_gcn_98_batch_normalization_792_moving_mean:f
Xassignvariableop_69_gcn_attention_49_core_gcn_98_batch_normalization_792_moving_variance:b
Tassignvariableop_70_gcn_attention_49_core_gcn_98_batch_normalization_793_moving_mean:f
Xassignvariableop_71_gcn_attention_49_core_gcn_98_batch_normalization_793_moving_variance:b
Tassignvariableop_72_gcn_attention_49_core_gcn_99_batch_normalization_794_moving_mean:f
Xassignvariableop_73_gcn_attention_49_core_gcn_99_batch_normalization_794_moving_variance:b
Tassignvariableop_74_gcn_attention_49_core_gcn_99_batch_normalization_795_moving_mean:f
Xassignvariableop_75_gcn_attention_49_core_gcn_99_batch_normalization_795_moving_variance:b
Tassignvariableop_76_gcn_attention_49_core_gcn_99_batch_normalization_796_moving_mean:f
Xassignvariableop_77_gcn_attention_49_core_gcn_99_batch_normalization_796_moving_variance:b
Tassignvariableop_78_gcn_attention_49_core_gcn_99_batch_normalization_797_moving_mean:f
Xassignvariableop_79_gcn_attention_49_core_gcn_99_batch_normalization_797_moving_variance:b
Tassignvariableop_80_gcn_attention_49_core_gcn_99_batch_normalization_798_moving_mean:f
Xassignvariableop_81_gcn_attention_49_core_gcn_99_batch_normalization_798_moving_variance:b
Tassignvariableop_82_gcn_attention_49_core_gcn_99_batch_normalization_799_moving_mean:f
Xassignvariableop_83_gcn_attention_49_core_gcn_99_batch_normalization_799_moving_variance:
identity_85ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_67ҐAssignVariableOp_68ҐAssignVariableOp_69ҐAssignVariableOp_7ҐAssignVariableOp_70ҐAssignVariableOp_71ҐAssignVariableOp_72ҐAssignVariableOp_73ҐAssignVariableOp_74ҐAssignVariableOp_75ҐAssignVariableOp_76ҐAssignVariableOp_77ҐAssignVariableOp_78ҐAssignVariableOp_79ҐAssignVariableOp_8ҐAssignVariableOp_80ҐAssignVariableOp_81ҐAssignVariableOp_82ҐAssignVariableOp_83ҐAssignVariableOp_9п/
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:U*
dtype0*ы.
valueс.Bо.UBCself_attention_layer/node0_Conv1D/kernel/.ATTRIBUTES/VARIABLE_VALUEBAself_attention_layer/node0_Conv1D/bias/.ATTRIBUTES/VARIABLE_VALUEBNself_attention_layer/node0_BatchNormalization/gamma/.ATTRIBUTES/VARIABLE_VALUEBMself_attention_layer/node0_BatchNormalization/beta/.ATTRIBUTES/VARIABLE_VALUEBTself_attention_layer/node0_BatchNormalization/moving_mean/.ATTRIBUTES/VARIABLE_VALUEBXself_attention_layer/node0_BatchNormalization/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBEself_attention_layer/distance_Dense/kernel/.ATTRIBUTES/VARIABLE_VALUEBMself_attention_layer/distance_Convolution2D/kernel/.ATTRIBUTES/VARIABLE_VALUEBKself_attention_layer/distance_Convolution2D/bias/.ATTRIBUTES/VARIABLE_VALUEBQself_attention_layer/distance_BatchNormalization/gamma/.ATTRIBUTES/VARIABLE_VALUEBPself_attention_layer/distance_BatchNormalization/beta/.ATTRIBUTES/VARIABLE_VALUEBWself_attention_layer/distance_BatchNormalization/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB[self_attention_layer/distance_BatchNormalization/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBDself_attention_layer/adj_Dense/embeddings/.ATTRIBUTES/VARIABLE_VALUEBNself_attention_layer/node_BatchNormalization5/gamma/.ATTRIBUTES/VARIABLE_VALUEBMself_attention_layer/node_BatchNormalization5/beta/.ATTRIBUTES/VARIABLE_VALUEBTself_attention_layer/node_BatchNormalization5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEBXself_attention_layer/node_BatchNormalization5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/node_Conv1D_out/kernel/.ATTRIBUTES/VARIABLE_VALUEBDself_attention_layer/node_Conv1D_out/bias/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/30/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/31/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/32/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/33/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/34/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/35/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/36/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/37/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/38/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/39/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/40/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/41/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/42/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/43/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/44/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/45/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/46/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/47/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/48/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/49/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/50/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/51/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/60/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/61/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/62/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/63/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/64/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/65/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/66/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/67/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/68/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/69/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/70/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/71/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/72/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/73/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/74/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/75/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/76/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/77/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/78/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/79/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/80/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/81/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/82/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/83/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesї
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:U*
dtype0*њ
valueµB≤UB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices„
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*к
_output_shapes„
‘:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*c
dtypesY
W2U2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity≤
AssignVariableOpAssignVariableOp3assignvariableop_gcn_attention_49_conv1d_392_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Є
AssignVariableOp_1AssignVariableOp3assignvariableop_1_gcn_attention_49_conv1d_392_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2∆
AssignVariableOp_2AssignVariableOpAassignvariableop_2_gcn_attention_49_batch_normalization_784_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3≈
AssignVariableOp_3AssignVariableOp@assignvariableop_3_gcn_attention_49_batch_normalization_784_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ћ
AssignVariableOp_4AssignVariableOpGassignvariableop_4_gcn_attention_49_batch_normalization_784_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5–
AssignVariableOp_5AssignVariableOpKassignvariableop_5_gcn_attention_49_batch_normalization_784_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Є
AssignVariableOp_6AssignVariableOp3assignvariableop_6_gcn_attention_49_dense_49_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ї
AssignVariableOp_7AssignVariableOp5assignvariableop_7_gcn_attention_49_conv2d_197_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Є
AssignVariableOp_8AssignVariableOp3assignvariableop_8_gcn_attention_49_conv2d_197_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9∆
AssignVariableOp_9AssignVariableOpAassignvariableop_9_gcn_attention_49_batch_normalization_786_gammaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10…
AssignVariableOp_10AssignVariableOpAassignvariableop_10_gcn_attention_49_batch_normalization_786_betaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11–
AssignVariableOp_11AssignVariableOpHassignvariableop_11_gcn_attention_49_batch_normalization_786_moving_meanIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12‘
AssignVariableOp_12AssignVariableOpLassignvariableop_12_gcn_attention_49_batch_normalization_786_moving_varianceIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ƒ
AssignVariableOp_13AssignVariableOp<assignvariableop_13_gcn_attention_49_embedding_49_embeddingsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14 
AssignVariableOp_14AssignVariableOpBassignvariableop_14_gcn_attention_49_batch_normalization_787_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15…
AssignVariableOp_15AssignVariableOpAassignvariableop_15_gcn_attention_49_batch_normalization_787_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16–
AssignVariableOp_16AssignVariableOpHassignvariableop_16_gcn_attention_49_batch_normalization_787_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17‘
AssignVariableOp_17AssignVariableOpLassignvariableop_17_gcn_attention_49_batch_normalization_787_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Њ
AssignVariableOp_18AssignVariableOp6assignvariableop_18_gcn_attention_49_conv1d_399_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Љ
AssignVariableOp_19AssignVariableOp4assignvariableop_19_gcn_attention_49_conv1d_399_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20 
AssignVariableOp_20AssignVariableOpBassignvariableop_20_gcn_attention_49_core_gcn_98_conv2d_198_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21»
AssignVariableOp_21AssignVariableOp@assignvariableop_21_gcn_attention_49_core_gcn_98_conv2d_198_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22÷
AssignVariableOp_22AssignVariableOpNassignvariableop_22_gcn_attention_49_core_gcn_98_batch_normalization_788_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23’
AssignVariableOp_23AssignVariableOpMassignvariableop_23_gcn_attention_49_core_gcn_98_batch_normalization_788_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24 
AssignVariableOp_24AssignVariableOpBassignvariableop_24_gcn_attention_49_core_gcn_98_conv1d_393_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25»
AssignVariableOp_25AssignVariableOp@assignvariableop_25_gcn_attention_49_core_gcn_98_conv1d_393_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26÷
AssignVariableOp_26AssignVariableOpNassignvariableop_26_gcn_attention_49_core_gcn_98_batch_normalization_789_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27’
AssignVariableOp_27AssignVariableOpMassignvariableop_27_gcn_attention_49_core_gcn_98_batch_normalization_789_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28 
AssignVariableOp_28AssignVariableOpBassignvariableop_28_gcn_attention_49_core_gcn_98_conv1d_394_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29»
AssignVariableOp_29AssignVariableOp@assignvariableop_29_gcn_attention_49_core_gcn_98_conv1d_394_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30÷
AssignVariableOp_30AssignVariableOpNassignvariableop_30_gcn_attention_49_core_gcn_98_batch_normalization_790_gammaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31’
AssignVariableOp_31AssignVariableOpMassignvariableop_31_gcn_attention_49_core_gcn_98_batch_normalization_790_betaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32 
AssignVariableOp_32AssignVariableOpBassignvariableop_32_gcn_attention_49_core_gcn_98_conv1d_395_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33»
AssignVariableOp_33AssignVariableOp@assignvariableop_33_gcn_attention_49_core_gcn_98_conv1d_395_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34÷
AssignVariableOp_34AssignVariableOpNassignvariableop_34_gcn_attention_49_core_gcn_98_batch_normalization_791_gammaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35’
AssignVariableOp_35AssignVariableOpMassignvariableop_35_gcn_attention_49_core_gcn_98_batch_normalization_791_betaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36÷
AssignVariableOp_36AssignVariableOpNassignvariableop_36_gcn_attention_49_core_gcn_98_batch_normalization_792_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37’
AssignVariableOp_37AssignVariableOpMassignvariableop_37_gcn_attention_49_core_gcn_98_batch_normalization_792_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38÷
AssignVariableOp_38AssignVariableOpNassignvariableop_38_gcn_attention_49_core_gcn_98_batch_normalization_793_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39’
AssignVariableOp_39AssignVariableOpMassignvariableop_39_gcn_attention_49_core_gcn_98_batch_normalization_793_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40 
AssignVariableOp_40AssignVariableOpBassignvariableop_40_gcn_attention_49_core_gcn_99_conv2d_199_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41»
AssignVariableOp_41AssignVariableOp@assignvariableop_41_gcn_attention_49_core_gcn_99_conv2d_199_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42÷
AssignVariableOp_42AssignVariableOpNassignvariableop_42_gcn_attention_49_core_gcn_99_batch_normalization_794_gammaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43’
AssignVariableOp_43AssignVariableOpMassignvariableop_43_gcn_attention_49_core_gcn_99_batch_normalization_794_betaIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44 
AssignVariableOp_44AssignVariableOpBassignvariableop_44_gcn_attention_49_core_gcn_99_conv1d_396_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45»
AssignVariableOp_45AssignVariableOp@assignvariableop_45_gcn_attention_49_core_gcn_99_conv1d_396_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46÷
AssignVariableOp_46AssignVariableOpNassignvariableop_46_gcn_attention_49_core_gcn_99_batch_normalization_795_gammaIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47’
AssignVariableOp_47AssignVariableOpMassignvariableop_47_gcn_attention_49_core_gcn_99_batch_normalization_795_betaIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48 
AssignVariableOp_48AssignVariableOpBassignvariableop_48_gcn_attention_49_core_gcn_99_conv1d_397_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49»
AssignVariableOp_49AssignVariableOp@assignvariableop_49_gcn_attention_49_core_gcn_99_conv1d_397_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50÷
AssignVariableOp_50AssignVariableOpNassignvariableop_50_gcn_attention_49_core_gcn_99_batch_normalization_796_gammaIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51’
AssignVariableOp_51AssignVariableOpMassignvariableop_51_gcn_attention_49_core_gcn_99_batch_normalization_796_betaIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52 
AssignVariableOp_52AssignVariableOpBassignvariableop_52_gcn_attention_49_core_gcn_99_conv1d_398_kernelIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53»
AssignVariableOp_53AssignVariableOp@assignvariableop_53_gcn_attention_49_core_gcn_99_conv1d_398_biasIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54÷
AssignVariableOp_54AssignVariableOpNassignvariableop_54_gcn_attention_49_core_gcn_99_batch_normalization_797_gammaIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55’
AssignVariableOp_55AssignVariableOpMassignvariableop_55_gcn_attention_49_core_gcn_99_batch_normalization_797_betaIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56÷
AssignVariableOp_56AssignVariableOpNassignvariableop_56_gcn_attention_49_core_gcn_99_batch_normalization_798_gammaIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57’
AssignVariableOp_57AssignVariableOpMassignvariableop_57_gcn_attention_49_core_gcn_99_batch_normalization_798_betaIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58÷
AssignVariableOp_58AssignVariableOpNassignvariableop_58_gcn_attention_49_core_gcn_99_batch_normalization_799_gammaIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59’
AssignVariableOp_59AssignVariableOpMassignvariableop_59_gcn_attention_49_core_gcn_99_batch_normalization_799_betaIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60№
AssignVariableOp_60AssignVariableOpTassignvariableop_60_gcn_attention_49_core_gcn_98_batch_normalization_788_moving_meanIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61а
AssignVariableOp_61AssignVariableOpXassignvariableop_61_gcn_attention_49_core_gcn_98_batch_normalization_788_moving_varianceIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62№
AssignVariableOp_62AssignVariableOpTassignvariableop_62_gcn_attention_49_core_gcn_98_batch_normalization_789_moving_meanIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63а
AssignVariableOp_63AssignVariableOpXassignvariableop_63_gcn_attention_49_core_gcn_98_batch_normalization_789_moving_varianceIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64№
AssignVariableOp_64AssignVariableOpTassignvariableop_64_gcn_attention_49_core_gcn_98_batch_normalization_790_moving_meanIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65а
AssignVariableOp_65AssignVariableOpXassignvariableop_65_gcn_attention_49_core_gcn_98_batch_normalization_790_moving_varianceIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66№
AssignVariableOp_66AssignVariableOpTassignvariableop_66_gcn_attention_49_core_gcn_98_batch_normalization_791_moving_meanIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67а
AssignVariableOp_67AssignVariableOpXassignvariableop_67_gcn_attention_49_core_gcn_98_batch_normalization_791_moving_varianceIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68№
AssignVariableOp_68AssignVariableOpTassignvariableop_68_gcn_attention_49_core_gcn_98_batch_normalization_792_moving_meanIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69а
AssignVariableOp_69AssignVariableOpXassignvariableop_69_gcn_attention_49_core_gcn_98_batch_normalization_792_moving_varianceIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70№
AssignVariableOp_70AssignVariableOpTassignvariableop_70_gcn_attention_49_core_gcn_98_batch_normalization_793_moving_meanIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71а
AssignVariableOp_71AssignVariableOpXassignvariableop_71_gcn_attention_49_core_gcn_98_batch_normalization_793_moving_varianceIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72№
AssignVariableOp_72AssignVariableOpTassignvariableop_72_gcn_attention_49_core_gcn_99_batch_normalization_794_moving_meanIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73а
AssignVariableOp_73AssignVariableOpXassignvariableop_73_gcn_attention_49_core_gcn_99_batch_normalization_794_moving_varianceIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74№
AssignVariableOp_74AssignVariableOpTassignvariableop_74_gcn_attention_49_core_gcn_99_batch_normalization_795_moving_meanIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75а
AssignVariableOp_75AssignVariableOpXassignvariableop_75_gcn_attention_49_core_gcn_99_batch_normalization_795_moving_varianceIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76№
AssignVariableOp_76AssignVariableOpTassignvariableop_76_gcn_attention_49_core_gcn_99_batch_normalization_796_moving_meanIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77а
AssignVariableOp_77AssignVariableOpXassignvariableop_77_gcn_attention_49_core_gcn_99_batch_normalization_796_moving_varianceIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78№
AssignVariableOp_78AssignVariableOpTassignvariableop_78_gcn_attention_49_core_gcn_99_batch_normalization_797_moving_meanIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79а
AssignVariableOp_79AssignVariableOpXassignvariableop_79_gcn_attention_49_core_gcn_99_batch_normalization_797_moving_varianceIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80№
AssignVariableOp_80AssignVariableOpTassignvariableop_80_gcn_attention_49_core_gcn_99_batch_normalization_798_moving_meanIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81а
AssignVariableOp_81AssignVariableOpXassignvariableop_81_gcn_attention_49_core_gcn_99_batch_normalization_798_moving_varianceIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82№
AssignVariableOp_82AssignVariableOpTassignvariableop_82_gcn_attention_49_core_gcn_99_batch_normalization_799_moving_meanIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83а
AssignVariableOp_83AssignVariableOpXassignvariableop_83_gcn_attention_49_core_gcn_99_batch_normalization_799_moving_varianceIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_839
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЦ
Identity_84Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_84f
Identity_85IdentityIdentity_84:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_85ю
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_85Identity_85:output:0*њ
_input_shapes≠
™: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
С	
”
8__inference_batch_normalization_795_layer_call_fn_614320

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_795_layer_call_and_return_conditional_losses_6127832
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Щ
¬
S__inference_batch_normalization_792_layer_call_and_return_conditional_losses_614135

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3¬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueќ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity№
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
р
Ж
S__inference_batch_normalization_796_layer_call_and_return_conditional_losses_612885

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identityИҐCast/ReadVariableOpҐCast_1/ReadVariableOpҐCast_2/ReadVariableOpҐCast_3/ReadVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityђ
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
С	
”
8__inference_batch_normalization_796_layer_call_fn_614400

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_796_layer_call_and_return_conditional_losses_6129452
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
р
Ж
S__inference_batch_normalization_789_layer_call_and_return_conditional_losses_613879

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identityИҐCast/ReadVariableOpҐCast_1/ReadVariableOpҐCast_2/ReadVariableOpҐCast_3/ReadVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityђ
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
У	
”
8__inference_batch_normalization_799_layer_call_fn_614609

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCall≠
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_799_layer_call_and_return_conditional_losses_6133352
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
У	
”
8__inference_batch_normalization_795_layer_call_fn_614307

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCall≠
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_795_layer_call_and_return_conditional_losses_6127232
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
р
Ж
S__inference_batch_normalization_795_layer_call_and_return_conditional_losses_612723

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identityИҐCast/ReadVariableOpҐCast_1/ReadVariableOpҐCast_2/ReadVariableOpҐCast_3/ReadVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityђ
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
С	
”
8__inference_batch_normalization_799_layer_call_fn_614622

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_799_layer_call_and_return_conditional_losses_6133952
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
У	
”
8__inference_batch_normalization_789_layer_call_fn_613846

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCall≠
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_789_layer_call_and_return_conditional_losses_6118232
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
≈	
”
8__inference_batch_normalization_786_layer_call_fn_613615

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityИҐStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_786_layer_call_and_return_conditional_losses_6114512
StatefulPartitionedCallХ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	
 
_user_specified_nameinputs
Щ
¬
S__inference_batch_normalization_794_layer_call_and_return_conditional_losses_612639

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3¬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueќ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity№
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
е
Ю
S__inference_batch_normalization_788_layer_call_and_return_conditional_losses_613815

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

IdentityЄ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
р
Ж
S__inference_batch_normalization_795_layer_call_and_return_conditional_losses_614340

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identityИҐCast/ReadVariableOpҐCast_1/ReadVariableOpҐCast_2/ReadVariableOpҐCast_3/ReadVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityђ
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
≈	
”
8__inference_batch_normalization_792_layer_call_fn_614099

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_792_layer_call_and_return_conditional_losses_6123512
StatefulPartitionedCallХ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
«	
”
8__inference_batch_normalization_794_layer_call_fn_614245

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_794_layer_call_and_return_conditional_losses_6125952
StatefulPartitionedCallХ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
У	
”
8__inference_batch_normalization_796_layer_call_fn_614387

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCall≠
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_796_layer_call_and_return_conditional_losses_6128852
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Џѕ
З:
__inference__traced_save_614964
file_prefixA
=savev2_gcn_attention_49_conv1d_392_kernel_read_readvariableop?
;savev2_gcn_attention_49_conv1d_392_bias_read_readvariableopM
Isavev2_gcn_attention_49_batch_normalization_784_gamma_read_readvariableopL
Hsavev2_gcn_attention_49_batch_normalization_784_beta_read_readvariableopS
Osavev2_gcn_attention_49_batch_normalization_784_moving_mean_read_readvariableopW
Ssavev2_gcn_attention_49_batch_normalization_784_moving_variance_read_readvariableop?
;savev2_gcn_attention_49_dense_49_kernel_read_readvariableopA
=savev2_gcn_attention_49_conv2d_197_kernel_read_readvariableop?
;savev2_gcn_attention_49_conv2d_197_bias_read_readvariableopM
Isavev2_gcn_attention_49_batch_normalization_786_gamma_read_readvariableopL
Hsavev2_gcn_attention_49_batch_normalization_786_beta_read_readvariableopS
Osavev2_gcn_attention_49_batch_normalization_786_moving_mean_read_readvariableopW
Ssavev2_gcn_attention_49_batch_normalization_786_moving_variance_read_readvariableopG
Csavev2_gcn_attention_49_embedding_49_embeddings_read_readvariableopM
Isavev2_gcn_attention_49_batch_normalization_787_gamma_read_readvariableopL
Hsavev2_gcn_attention_49_batch_normalization_787_beta_read_readvariableopS
Osavev2_gcn_attention_49_batch_normalization_787_moving_mean_read_readvariableopW
Ssavev2_gcn_attention_49_batch_normalization_787_moving_variance_read_readvariableopA
=savev2_gcn_attention_49_conv1d_399_kernel_read_readvariableop?
;savev2_gcn_attention_49_conv1d_399_bias_read_readvariableopM
Isavev2_gcn_attention_49_core_gcn_98_conv2d_198_kernel_read_readvariableopK
Gsavev2_gcn_attention_49_core_gcn_98_conv2d_198_bias_read_readvariableopY
Usavev2_gcn_attention_49_core_gcn_98_batch_normalization_788_gamma_read_readvariableopX
Tsavev2_gcn_attention_49_core_gcn_98_batch_normalization_788_beta_read_readvariableopM
Isavev2_gcn_attention_49_core_gcn_98_conv1d_393_kernel_read_readvariableopK
Gsavev2_gcn_attention_49_core_gcn_98_conv1d_393_bias_read_readvariableopY
Usavev2_gcn_attention_49_core_gcn_98_batch_normalization_789_gamma_read_readvariableopX
Tsavev2_gcn_attention_49_core_gcn_98_batch_normalization_789_beta_read_readvariableopM
Isavev2_gcn_attention_49_core_gcn_98_conv1d_394_kernel_read_readvariableopK
Gsavev2_gcn_attention_49_core_gcn_98_conv1d_394_bias_read_readvariableopY
Usavev2_gcn_attention_49_core_gcn_98_batch_normalization_790_gamma_read_readvariableopX
Tsavev2_gcn_attention_49_core_gcn_98_batch_normalization_790_beta_read_readvariableopM
Isavev2_gcn_attention_49_core_gcn_98_conv1d_395_kernel_read_readvariableopK
Gsavev2_gcn_attention_49_core_gcn_98_conv1d_395_bias_read_readvariableopY
Usavev2_gcn_attention_49_core_gcn_98_batch_normalization_791_gamma_read_readvariableopX
Tsavev2_gcn_attention_49_core_gcn_98_batch_normalization_791_beta_read_readvariableopY
Usavev2_gcn_attention_49_core_gcn_98_batch_normalization_792_gamma_read_readvariableopX
Tsavev2_gcn_attention_49_core_gcn_98_batch_normalization_792_beta_read_readvariableopY
Usavev2_gcn_attention_49_core_gcn_98_batch_normalization_793_gamma_read_readvariableopX
Tsavev2_gcn_attention_49_core_gcn_98_batch_normalization_793_beta_read_readvariableopM
Isavev2_gcn_attention_49_core_gcn_99_conv2d_199_kernel_read_readvariableopK
Gsavev2_gcn_attention_49_core_gcn_99_conv2d_199_bias_read_readvariableopY
Usavev2_gcn_attention_49_core_gcn_99_batch_normalization_794_gamma_read_readvariableopX
Tsavev2_gcn_attention_49_core_gcn_99_batch_normalization_794_beta_read_readvariableopM
Isavev2_gcn_attention_49_core_gcn_99_conv1d_396_kernel_read_readvariableopK
Gsavev2_gcn_attention_49_core_gcn_99_conv1d_396_bias_read_readvariableopY
Usavev2_gcn_attention_49_core_gcn_99_batch_normalization_795_gamma_read_readvariableopX
Tsavev2_gcn_attention_49_core_gcn_99_batch_normalization_795_beta_read_readvariableopM
Isavev2_gcn_attention_49_core_gcn_99_conv1d_397_kernel_read_readvariableopK
Gsavev2_gcn_attention_49_core_gcn_99_conv1d_397_bias_read_readvariableopY
Usavev2_gcn_attention_49_core_gcn_99_batch_normalization_796_gamma_read_readvariableopX
Tsavev2_gcn_attention_49_core_gcn_99_batch_normalization_796_beta_read_readvariableopM
Isavev2_gcn_attention_49_core_gcn_99_conv1d_398_kernel_read_readvariableopK
Gsavev2_gcn_attention_49_core_gcn_99_conv1d_398_bias_read_readvariableopY
Usavev2_gcn_attention_49_core_gcn_99_batch_normalization_797_gamma_read_readvariableopX
Tsavev2_gcn_attention_49_core_gcn_99_batch_normalization_797_beta_read_readvariableopY
Usavev2_gcn_attention_49_core_gcn_99_batch_normalization_798_gamma_read_readvariableopX
Tsavev2_gcn_attention_49_core_gcn_99_batch_normalization_798_beta_read_readvariableopY
Usavev2_gcn_attention_49_core_gcn_99_batch_normalization_799_gamma_read_readvariableopX
Tsavev2_gcn_attention_49_core_gcn_99_batch_normalization_799_beta_read_readvariableop_
[savev2_gcn_attention_49_core_gcn_98_batch_normalization_788_moving_mean_read_readvariableopc
_savev2_gcn_attention_49_core_gcn_98_batch_normalization_788_moving_variance_read_readvariableop_
[savev2_gcn_attention_49_core_gcn_98_batch_normalization_789_moving_mean_read_readvariableopc
_savev2_gcn_attention_49_core_gcn_98_batch_normalization_789_moving_variance_read_readvariableop_
[savev2_gcn_attention_49_core_gcn_98_batch_normalization_790_moving_mean_read_readvariableopc
_savev2_gcn_attention_49_core_gcn_98_batch_normalization_790_moving_variance_read_readvariableop_
[savev2_gcn_attention_49_core_gcn_98_batch_normalization_791_moving_mean_read_readvariableopc
_savev2_gcn_attention_49_core_gcn_98_batch_normalization_791_moving_variance_read_readvariableop_
[savev2_gcn_attention_49_core_gcn_98_batch_normalization_792_moving_mean_read_readvariableopc
_savev2_gcn_attention_49_core_gcn_98_batch_normalization_792_moving_variance_read_readvariableop_
[savev2_gcn_attention_49_core_gcn_98_batch_normalization_793_moving_mean_read_readvariableopc
_savev2_gcn_attention_49_core_gcn_98_batch_normalization_793_moving_variance_read_readvariableop_
[savev2_gcn_attention_49_core_gcn_99_batch_normalization_794_moving_mean_read_readvariableopc
_savev2_gcn_attention_49_core_gcn_99_batch_normalization_794_moving_variance_read_readvariableop_
[savev2_gcn_attention_49_core_gcn_99_batch_normalization_795_moving_mean_read_readvariableopc
_savev2_gcn_attention_49_core_gcn_99_batch_normalization_795_moving_variance_read_readvariableop_
[savev2_gcn_attention_49_core_gcn_99_batch_normalization_796_moving_mean_read_readvariableopc
_savev2_gcn_attention_49_core_gcn_99_batch_normalization_796_moving_variance_read_readvariableop_
[savev2_gcn_attention_49_core_gcn_99_batch_normalization_797_moving_mean_read_readvariableopc
_savev2_gcn_attention_49_core_gcn_99_batch_normalization_797_moving_variance_read_readvariableop_
[savev2_gcn_attention_49_core_gcn_99_batch_normalization_798_moving_mean_read_readvariableopc
_savev2_gcn_attention_49_core_gcn_99_batch_normalization_798_moving_variance_read_readvariableop_
[savev2_gcn_attention_49_core_gcn_99_batch_normalization_799_moving_mean_read_readvariableopc
_savev2_gcn_attention_49_core_gcn_99_batch_normalization_799_moving_variance_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameй/
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:U*
dtype0*ы.
valueс.Bо.UBCself_attention_layer/node0_Conv1D/kernel/.ATTRIBUTES/VARIABLE_VALUEBAself_attention_layer/node0_Conv1D/bias/.ATTRIBUTES/VARIABLE_VALUEBNself_attention_layer/node0_BatchNormalization/gamma/.ATTRIBUTES/VARIABLE_VALUEBMself_attention_layer/node0_BatchNormalization/beta/.ATTRIBUTES/VARIABLE_VALUEBTself_attention_layer/node0_BatchNormalization/moving_mean/.ATTRIBUTES/VARIABLE_VALUEBXself_attention_layer/node0_BatchNormalization/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBEself_attention_layer/distance_Dense/kernel/.ATTRIBUTES/VARIABLE_VALUEBMself_attention_layer/distance_Convolution2D/kernel/.ATTRIBUTES/VARIABLE_VALUEBKself_attention_layer/distance_Convolution2D/bias/.ATTRIBUTES/VARIABLE_VALUEBQself_attention_layer/distance_BatchNormalization/gamma/.ATTRIBUTES/VARIABLE_VALUEBPself_attention_layer/distance_BatchNormalization/beta/.ATTRIBUTES/VARIABLE_VALUEBWself_attention_layer/distance_BatchNormalization/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB[self_attention_layer/distance_BatchNormalization/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBDself_attention_layer/adj_Dense/embeddings/.ATTRIBUTES/VARIABLE_VALUEBNself_attention_layer/node_BatchNormalization5/gamma/.ATTRIBUTES/VARIABLE_VALUEBMself_attention_layer/node_BatchNormalization5/beta/.ATTRIBUTES/VARIABLE_VALUEBTself_attention_layer/node_BatchNormalization5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEBXself_attention_layer/node_BatchNormalization5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/node_Conv1D_out/kernel/.ATTRIBUTES/VARIABLE_VALUEBDself_attention_layer/node_Conv1D_out/bias/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/30/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/31/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/32/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/33/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/34/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/35/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/36/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/37/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/38/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/39/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/40/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/41/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/42/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/43/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/44/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/45/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/46/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/47/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/48/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/49/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/50/.ATTRIBUTES/VARIABLE_VALUEBFself_attention_layer/trainable_variables/51/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/60/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/61/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/62/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/63/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/64/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/65/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/66/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/67/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/68/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/69/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/70/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/71/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/72/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/73/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/74/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/75/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/76/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/77/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/78/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/79/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/80/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/81/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/82/.ATTRIBUTES/VARIABLE_VALUEB<self_attention_layer/variables/83/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesµ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:U*
dtype0*њ
valueµB≤UB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesў8
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0=savev2_gcn_attention_49_conv1d_392_kernel_read_readvariableop;savev2_gcn_attention_49_conv1d_392_bias_read_readvariableopIsavev2_gcn_attention_49_batch_normalization_784_gamma_read_readvariableopHsavev2_gcn_attention_49_batch_normalization_784_beta_read_readvariableopOsavev2_gcn_attention_49_batch_normalization_784_moving_mean_read_readvariableopSsavev2_gcn_attention_49_batch_normalization_784_moving_variance_read_readvariableop;savev2_gcn_attention_49_dense_49_kernel_read_readvariableop=savev2_gcn_attention_49_conv2d_197_kernel_read_readvariableop;savev2_gcn_attention_49_conv2d_197_bias_read_readvariableopIsavev2_gcn_attention_49_batch_normalization_786_gamma_read_readvariableopHsavev2_gcn_attention_49_batch_normalization_786_beta_read_readvariableopOsavev2_gcn_attention_49_batch_normalization_786_moving_mean_read_readvariableopSsavev2_gcn_attention_49_batch_normalization_786_moving_variance_read_readvariableopCsavev2_gcn_attention_49_embedding_49_embeddings_read_readvariableopIsavev2_gcn_attention_49_batch_normalization_787_gamma_read_readvariableopHsavev2_gcn_attention_49_batch_normalization_787_beta_read_readvariableopOsavev2_gcn_attention_49_batch_normalization_787_moving_mean_read_readvariableopSsavev2_gcn_attention_49_batch_normalization_787_moving_variance_read_readvariableop=savev2_gcn_attention_49_conv1d_399_kernel_read_readvariableop;savev2_gcn_attention_49_conv1d_399_bias_read_readvariableopIsavev2_gcn_attention_49_core_gcn_98_conv2d_198_kernel_read_readvariableopGsavev2_gcn_attention_49_core_gcn_98_conv2d_198_bias_read_readvariableopUsavev2_gcn_attention_49_core_gcn_98_batch_normalization_788_gamma_read_readvariableopTsavev2_gcn_attention_49_core_gcn_98_batch_normalization_788_beta_read_readvariableopIsavev2_gcn_attention_49_core_gcn_98_conv1d_393_kernel_read_readvariableopGsavev2_gcn_attention_49_core_gcn_98_conv1d_393_bias_read_readvariableopUsavev2_gcn_attention_49_core_gcn_98_batch_normalization_789_gamma_read_readvariableopTsavev2_gcn_attention_49_core_gcn_98_batch_normalization_789_beta_read_readvariableopIsavev2_gcn_attention_49_core_gcn_98_conv1d_394_kernel_read_readvariableopGsavev2_gcn_attention_49_core_gcn_98_conv1d_394_bias_read_readvariableopUsavev2_gcn_attention_49_core_gcn_98_batch_normalization_790_gamma_read_readvariableopTsavev2_gcn_attention_49_core_gcn_98_batch_normalization_790_beta_read_readvariableopIsavev2_gcn_attention_49_core_gcn_98_conv1d_395_kernel_read_readvariableopGsavev2_gcn_attention_49_core_gcn_98_conv1d_395_bias_read_readvariableopUsavev2_gcn_attention_49_core_gcn_98_batch_normalization_791_gamma_read_readvariableopTsavev2_gcn_attention_49_core_gcn_98_batch_normalization_791_beta_read_readvariableopUsavev2_gcn_attention_49_core_gcn_98_batch_normalization_792_gamma_read_readvariableopTsavev2_gcn_attention_49_core_gcn_98_batch_normalization_792_beta_read_readvariableopUsavev2_gcn_attention_49_core_gcn_98_batch_normalization_793_gamma_read_readvariableopTsavev2_gcn_attention_49_core_gcn_98_batch_normalization_793_beta_read_readvariableopIsavev2_gcn_attention_49_core_gcn_99_conv2d_199_kernel_read_readvariableopGsavev2_gcn_attention_49_core_gcn_99_conv2d_199_bias_read_readvariableopUsavev2_gcn_attention_49_core_gcn_99_batch_normalization_794_gamma_read_readvariableopTsavev2_gcn_attention_49_core_gcn_99_batch_normalization_794_beta_read_readvariableopIsavev2_gcn_attention_49_core_gcn_99_conv1d_396_kernel_read_readvariableopGsavev2_gcn_attention_49_core_gcn_99_conv1d_396_bias_read_readvariableopUsavev2_gcn_attention_49_core_gcn_99_batch_normalization_795_gamma_read_readvariableopTsavev2_gcn_attention_49_core_gcn_99_batch_normalization_795_beta_read_readvariableopIsavev2_gcn_attention_49_core_gcn_99_conv1d_397_kernel_read_readvariableopGsavev2_gcn_attention_49_core_gcn_99_conv1d_397_bias_read_readvariableopUsavev2_gcn_attention_49_core_gcn_99_batch_normalization_796_gamma_read_readvariableopTsavev2_gcn_attention_49_core_gcn_99_batch_normalization_796_beta_read_readvariableopIsavev2_gcn_attention_49_core_gcn_99_conv1d_398_kernel_read_readvariableopGsavev2_gcn_attention_49_core_gcn_99_conv1d_398_bias_read_readvariableopUsavev2_gcn_attention_49_core_gcn_99_batch_normalization_797_gamma_read_readvariableopTsavev2_gcn_attention_49_core_gcn_99_batch_normalization_797_beta_read_readvariableopUsavev2_gcn_attention_49_core_gcn_99_batch_normalization_798_gamma_read_readvariableopTsavev2_gcn_attention_49_core_gcn_99_batch_normalization_798_beta_read_readvariableopUsavev2_gcn_attention_49_core_gcn_99_batch_normalization_799_gamma_read_readvariableopTsavev2_gcn_attention_49_core_gcn_99_batch_normalization_799_beta_read_readvariableop[savev2_gcn_attention_49_core_gcn_98_batch_normalization_788_moving_mean_read_readvariableop_savev2_gcn_attention_49_core_gcn_98_batch_normalization_788_moving_variance_read_readvariableop[savev2_gcn_attention_49_core_gcn_98_batch_normalization_789_moving_mean_read_readvariableop_savev2_gcn_attention_49_core_gcn_98_batch_normalization_789_moving_variance_read_readvariableop[savev2_gcn_attention_49_core_gcn_98_batch_normalization_790_moving_mean_read_readvariableop_savev2_gcn_attention_49_core_gcn_98_batch_normalization_790_moving_variance_read_readvariableop[savev2_gcn_attention_49_core_gcn_98_batch_normalization_791_moving_mean_read_readvariableop_savev2_gcn_attention_49_core_gcn_98_batch_normalization_791_moving_variance_read_readvariableop[savev2_gcn_attention_49_core_gcn_98_batch_normalization_792_moving_mean_read_readvariableop_savev2_gcn_attention_49_core_gcn_98_batch_normalization_792_moving_variance_read_readvariableop[savev2_gcn_attention_49_core_gcn_98_batch_normalization_793_moving_mean_read_readvariableop_savev2_gcn_attention_49_core_gcn_98_batch_normalization_793_moving_variance_read_readvariableop[savev2_gcn_attention_49_core_gcn_99_batch_normalization_794_moving_mean_read_readvariableop_savev2_gcn_attention_49_core_gcn_99_batch_normalization_794_moving_variance_read_readvariableop[savev2_gcn_attention_49_core_gcn_99_batch_normalization_795_moving_mean_read_readvariableop_savev2_gcn_attention_49_core_gcn_99_batch_normalization_795_moving_variance_read_readvariableop[savev2_gcn_attention_49_core_gcn_99_batch_normalization_796_moving_mean_read_readvariableop_savev2_gcn_attention_49_core_gcn_99_batch_normalization_796_moving_variance_read_readvariableop[savev2_gcn_attention_49_core_gcn_99_batch_normalization_797_moving_mean_read_readvariableop_savev2_gcn_attention_49_core_gcn_99_batch_normalization_797_moving_variance_read_readvariableop[savev2_gcn_attention_49_core_gcn_99_batch_normalization_798_moving_mean_read_readvariableop_savev2_gcn_attention_49_core_gcn_99_batch_normalization_798_moving_variance_read_readvariableop[savev2_gcn_attention_49_core_gcn_99_batch_normalization_799_moving_mean_read_readvariableop_savev2_gcn_attention_49_core_gcn_99_batch_normalization_799_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *c
dtypesY
W2U2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*ю
_input_shapesм
й: :ђ::::::	:		:	:	:	:	:	:	::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_output_shapes
:ђ: 
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
е
Ю
S__inference_batch_normalization_786_layer_call_and_return_conditional_losses_613633

inputs%
readvariableop_resource:	'
readvariableop_1_resource:	6
(fusedbatchnormv3_readvariableop_resource:	8
*fusedbatchnormv3_readvariableop_1_resource:	
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:	*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:	*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	:	:	:	:	:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	2

IdentityЄ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	
 
_user_specified_nameinputs
≈	
”
8__inference_batch_normalization_798_layer_call_fn_614560

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_798_layer_call_and_return_conditional_losses_6132512
StatefulPartitionedCallХ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Щ
¬
S__inference_batch_normalization_798_layer_call_and_return_conditional_losses_613251

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3¬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueќ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity№
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
С	
”
8__inference_batch_normalization_793_layer_call_fn_614161

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_793_layer_call_and_return_conditional_losses_6124952
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
С	
”
8__inference_batch_normalization_797_layer_call_fn_614480

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_797_layer_call_and_return_conditional_losses_6131072
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
«	
”
8__inference_batch_normalization_786_layer_call_fn_613602

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityИҐStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_786_layer_call_and_return_conditional_losses_6114072
StatefulPartitionedCallХ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	
 
_user_specified_nameinputs
У	
”
8__inference_batch_normalization_787_layer_call_fn_613664

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCall≠
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_787_layer_call_and_return_conditional_losses_6115352
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ц
№
__inference_loss_fn_0_613742d
Mgcn_attention_49_conv1d_392_kernel_regularizer_square_readvariableop_resource:ђ
identityИҐDgcn_attention_49/conv1d_392/kernel/Regularizer/Square/ReadVariableOpЯ
Dgcn_attention_49/conv1d_392/kernel/Regularizer/Square/ReadVariableOpReadVariableOpMgcn_attention_49_conv1d_392_kernel_regularizer_square_readvariableop_resource*#
_output_shapes
:ђ*
dtype02F
Dgcn_attention_49/conv1d_392/kernel/Regularizer/Square/ReadVariableOpф
5gcn_attention_49/conv1d_392/kernel/Regularizer/SquareSquareLgcn_attention_49/conv1d_392/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:ђ27
5gcn_attention_49/conv1d_392/kernel/Regularizer/SquareЅ
4gcn_attention_49/conv1d_392/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          26
4gcn_attention_49/conv1d_392/kernel/Regularizer/ConstК
2gcn_attention_49/conv1d_392/kernel/Regularizer/SumSum9gcn_attention_49/conv1d_392/kernel/Regularizer/Square:y:0=gcn_attention_49/conv1d_392/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 24
2gcn_attention_49/conv1d_392/kernel/Regularizer/Sum±
4gcn_attention_49/conv1d_392/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—826
4gcn_attention_49/conv1d_392/kernel/Regularizer/mul/xМ
2gcn_attention_49/conv1d_392/kernel/Regularizer/mulMul=gcn_attention_49/conv1d_392/kernel/Regularizer/mul/x:output:0;gcn_attention_49/conv1d_392/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 24
2gcn_attention_49/conv1d_392/kernel/Regularizer/mulА
IdentityIdentity6gcn_attention_49/conv1d_392/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityХ
NoOpNoOpE^gcn_attention_49/conv1d_392/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2М
Dgcn_attention_49/conv1d_392/kernel/Regularizer/Square/ReadVariableOpDgcn_attention_49/conv1d_392/kernel/Regularizer/Square/ReadVariableOp
Щ
¬
S__inference_batch_normalization_786_layer_call_and_return_conditional_losses_611451

inputs%
readvariableop_resource:	'
readvariableop_1_resource:	6
(fusedbatchnormv3_readvariableop_resource:	8
*fusedbatchnormv3_readvariableop_1_resource:	
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:	*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:	*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	:	:	:	:	:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3¬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueќ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	2

Identity№
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	
 
_user_specified_nameinputs
р
Ж
S__inference_batch_normalization_793_layer_call_and_return_conditional_losses_612435

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identityИҐCast/ReadVariableOpҐCast_1/ReadVariableOpҐCast_2/ReadVariableOpҐCast_3/ReadVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityђ
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ*
‘
S__inference_batch_normalization_799_layer_call_and_return_conditional_losses_613395

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐCast/ReadVariableOpҐCast_1/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul…
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Г
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Щ
¬
S__inference_batch_normalization_788_layer_call_and_return_conditional_losses_611739

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3¬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueќ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity№
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ*
‘
S__inference_batch_normalization_797_layer_call_and_return_conditional_losses_613107

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐCast/ReadVariableOpҐCast_1/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul…
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Г
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
е
Ю
S__inference_batch_normalization_798_layer_call_and_return_conditional_losses_613207

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

IdentityЄ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
е
Ю
S__inference_batch_normalization_788_layer_call_and_return_conditional_losses_611695

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

IdentityЄ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
У	
”
8__inference_batch_normalization_797_layer_call_fn_614467

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCall≠
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_797_layer_call_and_return_conditional_losses_6130472
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ*
‘
S__inference_batch_normalization_789_layer_call_and_return_conditional_losses_613913

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐCast/ReadVariableOpҐCast_1/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul…
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Г
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
е
Ю
S__inference_batch_normalization_798_layer_call_and_return_conditional_losses_614578

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

IdentityЄ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Щ
¬
S__inference_batch_normalization_798_layer_call_and_return_conditional_losses_614596

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3¬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueќ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity№
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
«	
”
8__inference_batch_normalization_798_layer_call_fn_614547

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_798_layer_call_and_return_conditional_losses_6132072
StatefulPartitionedCallХ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
р
Ж
S__inference_batch_normalization_799_layer_call_and_return_conditional_losses_613335

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identityИҐCast/ReadVariableOpҐCast_1/ReadVariableOpҐCast_2/ReadVariableOpҐCast_3/ReadVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityђ
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ*
‘
S__inference_batch_normalization_796_layer_call_and_return_conditional_losses_614454

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐCast/ReadVariableOpҐCast_1/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul…
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Г
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ*
‘
S__inference_batch_normalization_795_layer_call_and_return_conditional_losses_612783

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐCast/ReadVariableOpҐCast_1/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul…
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Г
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Мж
ёn
__inference_tf_translate_611046
input_text1
input_text2
input_text3^
Ggcn_attention_49_conv1d_392_conv1d_expanddims_1_readvariableop_resource:ђI
;gcn_attention_49_conv1d_392_biasadd_readvariableop_resource:S
Egcn_attention_49_batch_normalization_784_cast_readvariableop_resource:U
Ggcn_attention_49_batch_normalization_784_cast_1_readvariableop_resource:U
Ggcn_attention_49_batch_normalization_784_cast_2_readvariableop_resource:U
Ggcn_attention_49_batch_normalization_784_cast_3_readvariableop_resource:M
;gcn_attention_49_dense_49_tensordot_readvariableop_resource:	T
:gcn_attention_49_conv2d_197_conv2d_readvariableop_resource:		I
;gcn_attention_49_conv2d_197_biasadd_readvariableop_resource:	N
@gcn_attention_49_batch_normalization_786_readvariableop_resource:	P
Bgcn_attention_49_batch_normalization_786_readvariableop_1_resource:	_
Qgcn_attention_49_batch_normalization_786_fusedbatchnormv3_readvariableop_resource:	a
Sgcn_attention_49_batch_normalization_786_fusedbatchnormv3_readvariableop_1_resource:	G
5gcn_attention_49_embedding_49_embedding_lookup_610384:	`
Fgcn_attention_49_core_gcn_98_conv2d_198_conv2d_readvariableop_resource:U
Ggcn_attention_49_core_gcn_98_conv2d_198_biasadd_readvariableop_resource:Z
Lgcn_attention_49_core_gcn_98_batch_normalization_788_readvariableop_resource:\
Ngcn_attention_49_core_gcn_98_batch_normalization_788_readvariableop_1_resource:k
]gcn_attention_49_core_gcn_98_batch_normalization_788_fusedbatchnormv3_readvariableop_resource:m
_gcn_attention_49_core_gcn_98_batch_normalization_788_fusedbatchnormv3_readvariableop_1_resource:i
Sgcn_attention_49_core_gcn_98_conv1d_393_conv1d_expanddims_1_readvariableop_resource:U
Ggcn_attention_49_core_gcn_98_conv1d_393_biasadd_readvariableop_resource:_
Qgcn_attention_49_core_gcn_98_batch_normalization_789_cast_readvariableop_resource:a
Sgcn_attention_49_core_gcn_98_batch_normalization_789_cast_1_readvariableop_resource:a
Sgcn_attention_49_core_gcn_98_batch_normalization_789_cast_2_readvariableop_resource:a
Sgcn_attention_49_core_gcn_98_batch_normalization_789_cast_3_readvariableop_resource:i
Sgcn_attention_49_core_gcn_98_conv1d_394_conv1d_expanddims_1_readvariableop_resource:U
Ggcn_attention_49_core_gcn_98_conv1d_394_biasadd_readvariableop_resource:_
Qgcn_attention_49_core_gcn_98_batch_normalization_790_cast_readvariableop_resource:a
Sgcn_attention_49_core_gcn_98_batch_normalization_790_cast_1_readvariableop_resource:a
Sgcn_attention_49_core_gcn_98_batch_normalization_790_cast_2_readvariableop_resource:a
Sgcn_attention_49_core_gcn_98_batch_normalization_790_cast_3_readvariableop_resource:i
Sgcn_attention_49_core_gcn_98_conv1d_395_conv1d_expanddims_1_readvariableop_resource:U
Ggcn_attention_49_core_gcn_98_conv1d_395_biasadd_readvariableop_resource:_
Qgcn_attention_49_core_gcn_98_batch_normalization_791_cast_readvariableop_resource:a
Sgcn_attention_49_core_gcn_98_batch_normalization_791_cast_1_readvariableop_resource:a
Sgcn_attention_49_core_gcn_98_batch_normalization_791_cast_2_readvariableop_resource:a
Sgcn_attention_49_core_gcn_98_batch_normalization_791_cast_3_readvariableop_resource:Z
Lgcn_attention_49_core_gcn_98_batch_normalization_792_readvariableop_resource:\
Ngcn_attention_49_core_gcn_98_batch_normalization_792_readvariableop_1_resource:k
]gcn_attention_49_core_gcn_98_batch_normalization_792_fusedbatchnormv3_readvariableop_resource:m
_gcn_attention_49_core_gcn_98_batch_normalization_792_fusedbatchnormv3_readvariableop_1_resource:_
Qgcn_attention_49_core_gcn_98_batch_normalization_793_cast_readvariableop_resource:a
Sgcn_attention_49_core_gcn_98_batch_normalization_793_cast_1_readvariableop_resource:a
Sgcn_attention_49_core_gcn_98_batch_normalization_793_cast_2_readvariableop_resource:a
Sgcn_attention_49_core_gcn_98_batch_normalization_793_cast_3_readvariableop_resource:`
Fgcn_attention_49_core_gcn_99_conv2d_199_conv2d_readvariableop_resource:U
Ggcn_attention_49_core_gcn_99_conv2d_199_biasadd_readvariableop_resource:Z
Lgcn_attention_49_core_gcn_99_batch_normalization_794_readvariableop_resource:\
Ngcn_attention_49_core_gcn_99_batch_normalization_794_readvariableop_1_resource:k
]gcn_attention_49_core_gcn_99_batch_normalization_794_fusedbatchnormv3_readvariableop_resource:m
_gcn_attention_49_core_gcn_99_batch_normalization_794_fusedbatchnormv3_readvariableop_1_resource:i
Sgcn_attention_49_core_gcn_99_conv1d_396_conv1d_expanddims_1_readvariableop_resource:U
Ggcn_attention_49_core_gcn_99_conv1d_396_biasadd_readvariableop_resource:_
Qgcn_attention_49_core_gcn_99_batch_normalization_795_cast_readvariableop_resource:a
Sgcn_attention_49_core_gcn_99_batch_normalization_795_cast_1_readvariableop_resource:a
Sgcn_attention_49_core_gcn_99_batch_normalization_795_cast_2_readvariableop_resource:a
Sgcn_attention_49_core_gcn_99_batch_normalization_795_cast_3_readvariableop_resource:i
Sgcn_attention_49_core_gcn_99_conv1d_397_conv1d_expanddims_1_readvariableop_resource:U
Ggcn_attention_49_core_gcn_99_conv1d_397_biasadd_readvariableop_resource:_
Qgcn_attention_49_core_gcn_99_batch_normalization_796_cast_readvariableop_resource:a
Sgcn_attention_49_core_gcn_99_batch_normalization_796_cast_1_readvariableop_resource:a
Sgcn_attention_49_core_gcn_99_batch_normalization_796_cast_2_readvariableop_resource:a
Sgcn_attention_49_core_gcn_99_batch_normalization_796_cast_3_readvariableop_resource:i
Sgcn_attention_49_core_gcn_99_conv1d_398_conv1d_expanddims_1_readvariableop_resource:U
Ggcn_attention_49_core_gcn_99_conv1d_398_biasadd_readvariableop_resource:_
Qgcn_attention_49_core_gcn_99_batch_normalization_797_cast_readvariableop_resource:a
Sgcn_attention_49_core_gcn_99_batch_normalization_797_cast_1_readvariableop_resource:a
Sgcn_attention_49_core_gcn_99_batch_normalization_797_cast_2_readvariableop_resource:a
Sgcn_attention_49_core_gcn_99_batch_normalization_797_cast_3_readvariableop_resource:Z
Lgcn_attention_49_core_gcn_99_batch_normalization_798_readvariableop_resource:\
Ngcn_attention_49_core_gcn_99_batch_normalization_798_readvariableop_1_resource:k
]gcn_attention_49_core_gcn_99_batch_normalization_798_fusedbatchnormv3_readvariableop_resource:m
_gcn_attention_49_core_gcn_99_batch_normalization_798_fusedbatchnormv3_readvariableop_1_resource:_
Qgcn_attention_49_core_gcn_99_batch_normalization_799_cast_readvariableop_resource:a
Sgcn_attention_49_core_gcn_99_batch_normalization_799_cast_1_readvariableop_resource:a
Sgcn_attention_49_core_gcn_99_batch_normalization_799_cast_2_readvariableop_resource:a
Sgcn_attention_49_core_gcn_99_batch_normalization_799_cast_3_readvariableop_resource:S
Egcn_attention_49_batch_normalization_787_cast_readvariableop_resource:U
Ggcn_attention_49_batch_normalization_787_cast_1_readvariableop_resource:U
Ggcn_attention_49_batch_normalization_787_cast_2_readvariableop_resource:U
Ggcn_attention_49_batch_normalization_787_cast_3_readvariableop_resource:]
Ggcn_attention_49_conv1d_399_conv1d_expanddims_1_readvariableop_resource:I
;gcn_attention_49_conv1d_399_biasadd_readvariableop_resource:
identityИҐ<gcn_attention_49/batch_normalization_784/Cast/ReadVariableOpҐ>gcn_attention_49/batch_normalization_784/Cast_1/ReadVariableOpҐ>gcn_attention_49/batch_normalization_784/Cast_2/ReadVariableOpҐ>gcn_attention_49/batch_normalization_784/Cast_3/ReadVariableOpҐHgcn_attention_49/batch_normalization_786/FusedBatchNormV3/ReadVariableOpҐJgcn_attention_49/batch_normalization_786/FusedBatchNormV3/ReadVariableOp_1Ґ7gcn_attention_49/batch_normalization_786/ReadVariableOpҐ9gcn_attention_49/batch_normalization_786/ReadVariableOp_1Ґ<gcn_attention_49/batch_normalization_787/Cast/ReadVariableOpҐ>gcn_attention_49/batch_normalization_787/Cast_1/ReadVariableOpҐ>gcn_attention_49/batch_normalization_787/Cast_2/ReadVariableOpҐ>gcn_attention_49/batch_normalization_787/Cast_3/ReadVariableOpҐ2gcn_attention_49/conv1d_392/BiasAdd/ReadVariableOpҐ>gcn_attention_49/conv1d_392/conv1d/ExpandDims_1/ReadVariableOpҐ2gcn_attention_49/conv1d_399/BiasAdd/ReadVariableOpҐ>gcn_attention_49/conv1d_399/conv1d/ExpandDims_1/ReadVariableOpҐ2gcn_attention_49/conv2d_197/BiasAdd/ReadVariableOpҐ1gcn_attention_49/conv2d_197/Conv2D/ReadVariableOpҐTgcn_attention_49/core_gcn_98/batch_normalization_788/FusedBatchNormV3/ReadVariableOpҐVgcn_attention_49/core_gcn_98/batch_normalization_788/FusedBatchNormV3/ReadVariableOp_1ҐCgcn_attention_49/core_gcn_98/batch_normalization_788/ReadVariableOpҐEgcn_attention_49/core_gcn_98/batch_normalization_788/ReadVariableOp_1ҐHgcn_attention_49/core_gcn_98/batch_normalization_789/Cast/ReadVariableOpҐJgcn_attention_49/core_gcn_98/batch_normalization_789/Cast_1/ReadVariableOpҐJgcn_attention_49/core_gcn_98/batch_normalization_789/Cast_2/ReadVariableOpҐJgcn_attention_49/core_gcn_98/batch_normalization_789/Cast_3/ReadVariableOpҐHgcn_attention_49/core_gcn_98/batch_normalization_790/Cast/ReadVariableOpҐJgcn_attention_49/core_gcn_98/batch_normalization_790/Cast_1/ReadVariableOpҐJgcn_attention_49/core_gcn_98/batch_normalization_790/Cast_2/ReadVariableOpҐJgcn_attention_49/core_gcn_98/batch_normalization_790/Cast_3/ReadVariableOpҐHgcn_attention_49/core_gcn_98/batch_normalization_791/Cast/ReadVariableOpҐJgcn_attention_49/core_gcn_98/batch_normalization_791/Cast_1/ReadVariableOpҐJgcn_attention_49/core_gcn_98/batch_normalization_791/Cast_2/ReadVariableOpҐJgcn_attention_49/core_gcn_98/batch_normalization_791/Cast_3/ReadVariableOpҐTgcn_attention_49/core_gcn_98/batch_normalization_792/FusedBatchNormV3/ReadVariableOpҐVgcn_attention_49/core_gcn_98/batch_normalization_792/FusedBatchNormV3/ReadVariableOp_1ҐCgcn_attention_49/core_gcn_98/batch_normalization_792/ReadVariableOpҐEgcn_attention_49/core_gcn_98/batch_normalization_792/ReadVariableOp_1ҐHgcn_attention_49/core_gcn_98/batch_normalization_793/Cast/ReadVariableOpҐJgcn_attention_49/core_gcn_98/batch_normalization_793/Cast_1/ReadVariableOpҐJgcn_attention_49/core_gcn_98/batch_normalization_793/Cast_2/ReadVariableOpҐJgcn_attention_49/core_gcn_98/batch_normalization_793/Cast_3/ReadVariableOpҐ>gcn_attention_49/core_gcn_98/conv1d_393/BiasAdd/ReadVariableOpҐJgcn_attention_49/core_gcn_98/conv1d_393/conv1d/ExpandDims_1/ReadVariableOpҐ>gcn_attention_49/core_gcn_98/conv1d_394/BiasAdd/ReadVariableOpҐJgcn_attention_49/core_gcn_98/conv1d_394/conv1d/ExpandDims_1/ReadVariableOpҐ>gcn_attention_49/core_gcn_98/conv1d_395/BiasAdd/ReadVariableOpҐJgcn_attention_49/core_gcn_98/conv1d_395/conv1d/ExpandDims_1/ReadVariableOpҐ>gcn_attention_49/core_gcn_98/conv2d_198/BiasAdd/ReadVariableOpҐ=gcn_attention_49/core_gcn_98/conv2d_198/Conv2D/ReadVariableOpҐTgcn_attention_49/core_gcn_99/batch_normalization_794/FusedBatchNormV3/ReadVariableOpҐVgcn_attention_49/core_gcn_99/batch_normalization_794/FusedBatchNormV3/ReadVariableOp_1ҐCgcn_attention_49/core_gcn_99/batch_normalization_794/ReadVariableOpҐEgcn_attention_49/core_gcn_99/batch_normalization_794/ReadVariableOp_1ҐHgcn_attention_49/core_gcn_99/batch_normalization_795/Cast/ReadVariableOpҐJgcn_attention_49/core_gcn_99/batch_normalization_795/Cast_1/ReadVariableOpҐJgcn_attention_49/core_gcn_99/batch_normalization_795/Cast_2/ReadVariableOpҐJgcn_attention_49/core_gcn_99/batch_normalization_795/Cast_3/ReadVariableOpҐHgcn_attention_49/core_gcn_99/batch_normalization_796/Cast/ReadVariableOpҐJgcn_attention_49/core_gcn_99/batch_normalization_796/Cast_1/ReadVariableOpҐJgcn_attention_49/core_gcn_99/batch_normalization_796/Cast_2/ReadVariableOpҐJgcn_attention_49/core_gcn_99/batch_normalization_796/Cast_3/ReadVariableOpҐHgcn_attention_49/core_gcn_99/batch_normalization_797/Cast/ReadVariableOpҐJgcn_attention_49/core_gcn_99/batch_normalization_797/Cast_1/ReadVariableOpҐJgcn_attention_49/core_gcn_99/batch_normalization_797/Cast_2/ReadVariableOpҐJgcn_attention_49/core_gcn_99/batch_normalization_797/Cast_3/ReadVariableOpҐTgcn_attention_49/core_gcn_99/batch_normalization_798/FusedBatchNormV3/ReadVariableOpҐVgcn_attention_49/core_gcn_99/batch_normalization_798/FusedBatchNormV3/ReadVariableOp_1ҐCgcn_attention_49/core_gcn_99/batch_normalization_798/ReadVariableOpҐEgcn_attention_49/core_gcn_99/batch_normalization_798/ReadVariableOp_1ҐHgcn_attention_49/core_gcn_99/batch_normalization_799/Cast/ReadVariableOpҐJgcn_attention_49/core_gcn_99/batch_normalization_799/Cast_1/ReadVariableOpҐJgcn_attention_49/core_gcn_99/batch_normalization_799/Cast_2/ReadVariableOpҐJgcn_attention_49/core_gcn_99/batch_normalization_799/Cast_3/ReadVariableOpҐ>gcn_attention_49/core_gcn_99/conv1d_396/BiasAdd/ReadVariableOpҐJgcn_attention_49/core_gcn_99/conv1d_396/conv1d/ExpandDims_1/ReadVariableOpҐ>gcn_attention_49/core_gcn_99/conv1d_397/BiasAdd/ReadVariableOpҐJgcn_attention_49/core_gcn_99/conv1d_397/conv1d/ExpandDims_1/ReadVariableOpҐ>gcn_attention_49/core_gcn_99/conv1d_398/BiasAdd/ReadVariableOpҐJgcn_attention_49/core_gcn_99/conv1d_398/conv1d/ExpandDims_1/ReadVariableOpҐ>gcn_attention_49/core_gcn_99/conv2d_199/BiasAdd/ReadVariableOpҐ=gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/ReadVariableOpҐ2gcn_attention_49/dense_49/Tensordot/ReadVariableOpҐ.gcn_attention_49/embedding_49/embedding_lookup±
1gcn_attention_49/conv1d_392/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€23
1gcn_attention_49/conv1d_392/conv1d/ExpandDims/dimщ
-gcn_attention_49/conv1d_392/conv1d/ExpandDims
ExpandDimsinput_text1:gcn_attention_49/conv1d_392/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#€€€€€€€€€€€€€€€€€€ђ2/
-gcn_attention_49/conv1d_392/conv1d/ExpandDimsН
>gcn_attention_49/conv1d_392/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpGgcn_attention_49_conv1d_392_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:ђ*
dtype02@
>gcn_attention_49/conv1d_392/conv1d/ExpandDims_1/ReadVariableOpђ
3gcn_attention_49/conv1d_392/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3gcn_attention_49/conv1d_392/conv1d/ExpandDims_1/dim®
/gcn_attention_49/conv1d_392/conv1d/ExpandDims_1
ExpandDimsFgcn_attention_49/conv1d_392/conv1d/ExpandDims_1/ReadVariableOp:value:0<gcn_attention_49/conv1d_392/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:ђ21
/gcn_attention_49/conv1d_392/conv1d/ExpandDims_1ѓ
"gcn_attention_49/conv1d_392/conv1dConv2D6gcn_attention_49/conv1d_392/conv1d/ExpandDims:output:08gcn_attention_49/conv1d_392/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
2$
"gcn_attention_49/conv1d_392/conv1dп
*gcn_attention_49/conv1d_392/conv1d/SqueezeSqueeze+gcn_attention_49/conv1d_392/conv1d:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
squeeze_dims

э€€€€€€€€2,
*gcn_attention_49/conv1d_392/conv1d/Squeezeа
2gcn_attention_49/conv1d_392/BiasAdd/ReadVariableOpReadVariableOp;gcn_attention_49_conv1d_392_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2gcn_attention_49/conv1d_392/BiasAdd/ReadVariableOpЕ
#gcn_attention_49/conv1d_392/BiasAddBiasAdd3gcn_attention_49/conv1d_392/conv1d/Squeeze:output:0:gcn_attention_49/conv1d_392/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2%
#gcn_attention_49/conv1d_392/BiasAddЅ
$gcn_attention_49/activation_245/ReluRelu,gcn_attention_49/conv1d_392/BiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2&
$gcn_attention_49/activation_245/Reluю
<gcn_attention_49/batch_normalization_784/Cast/ReadVariableOpReadVariableOpEgcn_attention_49_batch_normalization_784_cast_readvariableop_resource*
_output_shapes
:*
dtype02>
<gcn_attention_49/batch_normalization_784/Cast/ReadVariableOpД
>gcn_attention_49/batch_normalization_784/Cast_1/ReadVariableOpReadVariableOpGgcn_attention_49_batch_normalization_784_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_49/batch_normalization_784/Cast_1/ReadVariableOpД
>gcn_attention_49/batch_normalization_784/Cast_2/ReadVariableOpReadVariableOpGgcn_attention_49_batch_normalization_784_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_49/batch_normalization_784/Cast_2/ReadVariableOpД
>gcn_attention_49/batch_normalization_784/Cast_3/ReadVariableOpReadVariableOpGgcn_attention_49_batch_normalization_784_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_49/batch_normalization_784/Cast_3/ReadVariableOpє
8gcn_attention_49/batch_normalization_784/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2:
8gcn_attention_49/batch_normalization_784/batchnorm/add/y©
6gcn_attention_49/batch_normalization_784/batchnorm/addAddV2Fgcn_attention_49/batch_normalization_784/Cast_1/ReadVariableOp:value:0Agcn_attention_49/batch_normalization_784/batchnorm/add/y:output:0*
T0*
_output_shapes
:28
6gcn_attention_49/batch_normalization_784/batchnorm/addё
8gcn_attention_49/batch_normalization_784/batchnorm/RsqrtRsqrt:gcn_attention_49/batch_normalization_784/batchnorm/add:z:0*
T0*
_output_shapes
:2:
8gcn_attention_49/batch_normalization_784/batchnorm/RsqrtҐ
6gcn_attention_49/batch_normalization_784/batchnorm/mulMul<gcn_attention_49/batch_normalization_784/batchnorm/Rsqrt:y:0Fgcn_attention_49/batch_normalization_784/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:28
6gcn_attention_49/batch_normalization_784/batchnorm/mul™
8gcn_attention_49/batch_normalization_784/batchnorm/mul_1Mul2gcn_attention_49/activation_245/Relu:activations:0:gcn_attention_49/batch_normalization_784/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2:
8gcn_attention_49/batch_normalization_784/batchnorm/mul_1Ґ
8gcn_attention_49/batch_normalization_784/batchnorm/mul_2MulDgcn_attention_49/batch_normalization_784/Cast/ReadVariableOp:value:0:gcn_attention_49/batch_normalization_784/batchnorm/mul:z:0*
T0*
_output_shapes
:2:
8gcn_attention_49/batch_normalization_784/batchnorm/mul_2Ґ
6gcn_attention_49/batch_normalization_784/batchnorm/subSubFgcn_attention_49/batch_normalization_784/Cast_2/ReadVariableOp:value:0<gcn_attention_49/batch_normalization_784/batchnorm/mul_2:z:0*
T0*
_output_shapes
:28
6gcn_attention_49/batch_normalization_784/batchnorm/subґ
8gcn_attention_49/batch_normalization_784/batchnorm/add_1AddV2<gcn_attention_49/batch_normalization_784/batchnorm/mul_1:z:0:gcn_attention_49/batch_normalization_784/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2:
8gcn_attention_49/batch_normalization_784/batchnorm/add_1„
%gcn_attention_49/dropout_588/IdentityIdentity<gcn_attention_49/batch_normalization_784/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2'
%gcn_attention_49/dropout_588/Identityд
2gcn_attention_49/dense_49/Tensordot/ReadVariableOpReadVariableOp;gcn_attention_49_dense_49_tensordot_readvariableop_resource*
_output_shapes

:	*
dtype024
2gcn_attention_49/dense_49/Tensordot/ReadVariableOpЮ
(gcn_attention_49/dense_49/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2*
(gcn_attention_49/dense_49/Tensordot/axes©
(gcn_attention_49/dense_49/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(gcn_attention_49/dense_49/Tensordot/freeС
)gcn_attention_49/dense_49/Tensordot/ShapeShapeinput_text3*
T0*
_output_shapes
:2+
)gcn_attention_49/dense_49/Tensordot/Shape®
1gcn_attention_49/dense_49/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1gcn_attention_49/dense_49/Tensordot/GatherV2/axis”
,gcn_attention_49/dense_49/Tensordot/GatherV2GatherV22gcn_attention_49/dense_49/Tensordot/Shape:output:01gcn_attention_49/dense_49/Tensordot/free:output:0:gcn_attention_49/dense_49/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,gcn_attention_49/dense_49/Tensordot/GatherV2ђ
3gcn_attention_49/dense_49/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 25
3gcn_attention_49/dense_49/Tensordot/GatherV2_1/axisў
.gcn_attention_49/dense_49/Tensordot/GatherV2_1GatherV22gcn_attention_49/dense_49/Tensordot/Shape:output:01gcn_attention_49/dense_49/Tensordot/axes:output:0<gcn_attention_49/dense_49/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:20
.gcn_attention_49/dense_49/Tensordot/GatherV2_1†
)gcn_attention_49/dense_49/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)gcn_attention_49/dense_49/Tensordot/Constи
(gcn_attention_49/dense_49/Tensordot/ProdProd5gcn_attention_49/dense_49/Tensordot/GatherV2:output:02gcn_attention_49/dense_49/Tensordot/Const:output:0*
T0*
_output_shapes
: 2*
(gcn_attention_49/dense_49/Tensordot/Prod§
+gcn_attention_49/dense_49/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+gcn_attention_49/dense_49/Tensordot/Const_1р
*gcn_attention_49/dense_49/Tensordot/Prod_1Prod7gcn_attention_49/dense_49/Tensordot/GatherV2_1:output:04gcn_attention_49/dense_49/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2,
*gcn_attention_49/dense_49/Tensordot/Prod_1§
/gcn_attention_49/dense_49/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/gcn_attention_49/dense_49/Tensordot/concat/axis≤
*gcn_attention_49/dense_49/Tensordot/concatConcatV21gcn_attention_49/dense_49/Tensordot/free:output:01gcn_attention_49/dense_49/Tensordot/axes:output:08gcn_attention_49/dense_49/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*gcn_attention_49/dense_49/Tensordot/concatф
)gcn_attention_49/dense_49/Tensordot/stackPack1gcn_attention_49/dense_49/Tensordot/Prod:output:03gcn_attention_49/dense_49/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2+
)gcn_attention_49/dense_49/Tensordot/stackщ
-gcn_attention_49/dense_49/Tensordot/transpose	Transposeinput_text33gcn_attention_49/dense_49/Tensordot/concat:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2/
-gcn_attention_49/dense_49/Tensordot/transposeЗ
+gcn_attention_49/dense_49/Tensordot/ReshapeReshape1gcn_attention_49/dense_49/Tensordot/transpose:y:02gcn_attention_49/dense_49/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2-
+gcn_attention_49/dense_49/Tensordot/ReshapeЖ
*gcn_attention_49/dense_49/Tensordot/MatMulMatMul4gcn_attention_49/dense_49/Tensordot/Reshape:output:0:gcn_attention_49/dense_49/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€	2,
*gcn_attention_49/dense_49/Tensordot/MatMul§
+gcn_attention_49/dense_49/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	2-
+gcn_attention_49/dense_49/Tensordot/Const_2®
1gcn_attention_49/dense_49/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1gcn_attention_49/dense_49/Tensordot/concat_1/axisњ
,gcn_attention_49/dense_49/Tensordot/concat_1ConcatV25gcn_attention_49/dense_49/Tensordot/GatherV2:output:04gcn_attention_49/dense_49/Tensordot/Const_2:output:0:gcn_attention_49/dense_49/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2.
,gcn_attention_49/dense_49/Tensordot/concat_1О
#gcn_attention_49/dense_49/TensordotReshape4gcn_attention_49/dense_49/Tensordot/MatMul:product:05gcn_attention_49/dense_49/Tensordot/concat_1:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	2%
#gcn_attention_49/dense_49/Tensordotй
1gcn_attention_49/conv2d_197/Conv2D/ReadVariableOpReadVariableOp:gcn_attention_49_conv2d_197_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype023
1gcn_attention_49/conv2d_197/Conv2D/ReadVariableOpѓ
"gcn_attention_49/conv2d_197/Conv2DConv2D,gcn_attention_49/dense_49/Tensordot:output:09gcn_attention_49/conv2d_197/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	*
paddingSAME*
strides
2$
"gcn_attention_49/conv2d_197/Conv2Dа
2gcn_attention_49/conv2d_197/BiasAdd/ReadVariableOpReadVariableOp;gcn_attention_49_conv2d_197_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype024
2gcn_attention_49/conv2d_197/BiasAdd/ReadVariableOpК
#gcn_attention_49/conv2d_197/BiasAddBiasAdd+gcn_attention_49/conv2d_197/Conv2D:output:0:gcn_attention_49/conv2d_197/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	2%
#gcn_attention_49/conv2d_197/BiasAddќ
$gcn_attention_49/activation_247/ReluRelu,gcn_attention_49/conv2d_197/BiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	2&
$gcn_attention_49/activation_247/Reluп
7gcn_attention_49/batch_normalization_786/ReadVariableOpReadVariableOp@gcn_attention_49_batch_normalization_786_readvariableop_resource*
_output_shapes
:	*
dtype029
7gcn_attention_49/batch_normalization_786/ReadVariableOpх
9gcn_attention_49/batch_normalization_786/ReadVariableOp_1ReadVariableOpBgcn_attention_49_batch_normalization_786_readvariableop_1_resource*
_output_shapes
:	*
dtype02;
9gcn_attention_49/batch_normalization_786/ReadVariableOp_1Ґ
Hgcn_attention_49/batch_normalization_786/FusedBatchNormV3/ReadVariableOpReadVariableOpQgcn_attention_49_batch_normalization_786_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:	*
dtype02J
Hgcn_attention_49/batch_normalization_786/FusedBatchNormV3/ReadVariableOp®
Jgcn_attention_49/batch_normalization_786/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpSgcn_attention_49_batch_normalization_786_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:	*
dtype02L
Jgcn_attention_49/batch_normalization_786/FusedBatchNormV3/ReadVariableOp_1ю
9gcn_attention_49/batch_normalization_786/FusedBatchNormV3FusedBatchNormV32gcn_attention_49/activation_247/Relu:activations:0?gcn_attention_49/batch_normalization_786/ReadVariableOp:value:0Agcn_attention_49/batch_normalization_786/ReadVariableOp_1:value:0Pgcn_attention_49/batch_normalization_786/FusedBatchNormV3/ReadVariableOp:value:0Rgcn_attention_49/batch_normalization_786/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	:	:	:	:	:*
epsilon%oГ:*
is_training( 2;
9gcn_attention_49/batch_normalization_786/FusedBatchNormV3е
%gcn_attention_49/dropout_590/IdentityIdentity=gcn_attention_49/batch_normalization_786/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	2'
%gcn_attention_49/dropout_590/Identityі
gcn_attention_49/SqueezeSqueezeinput_text2*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€*
squeeze_dims

€€€€€€€€€2
gcn_attention_49/Squeeze 
"gcn_attention_49/embedding_49/CastCast!gcn_attention_49/Squeeze:output:0*

DstT0*

SrcT0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2$
"gcn_attention_49/embedding_49/Cast©
.gcn_attention_49/embedding_49/embedding_lookupResourceGather5gcn_attention_49_embedding_49_embedding_lookup_610384&gcn_attention_49/embedding_49/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*H
_class>
<:loc:@gcn_attention_49/embedding_49/embedding_lookup/610384*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	*
dtype020
.gcn_attention_49/embedding_49/embedding_lookupы
7gcn_attention_49/embedding_49/embedding_lookup/IdentityIdentity7gcn_attention_49/embedding_49/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*H
_class>
<:loc:@gcn_attention_49/embedding_49/embedding_lookup/610384*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	29
7gcn_attention_49/embedding_49/embedding_lookup/IdentityР
9gcn_attention_49/embedding_49/embedding_lookup/Identity_1Identity@gcn_attention_49/embedding_49/embedding_lookup/Identity:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	2;
9gcn_attention_49/embedding_49/embedding_lookup/Identity_1Ц
(gcn_attention_49/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2*
(gcn_attention_49/concatenate/concat/axis“
#gcn_attention_49/concatenate/concatConcatV2.gcn_attention_49/dropout_590/Identity:output:0Bgcn_attention_49/embedding_49/embedding_lookup/Identity_1:output:01gcn_attention_49/concatenate/concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2%
#gcn_attention_49/concatenate/concatН
=gcn_attention_49/core_gcn_98/conv2d_198/Conv2D/ReadVariableOpReadVariableOpFgcn_attention_49_core_gcn_98_conv2d_198_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02?
=gcn_attention_49/core_gcn_98/conv2d_198/Conv2D/ReadVariableOp”
.gcn_attention_49/core_gcn_98/conv2d_198/Conv2DConv2D,gcn_attention_49/concatenate/concat:output:0Egcn_attention_49/core_gcn_98/conv2d_198/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
20
.gcn_attention_49/core_gcn_98/conv2d_198/Conv2DД
>gcn_attention_49/core_gcn_98/conv2d_198/BiasAdd/ReadVariableOpReadVariableOpGgcn_attention_49_core_gcn_98_conv2d_198_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_49/core_gcn_98/conv2d_198/BiasAdd/ReadVariableOpЇ
/gcn_attention_49/core_gcn_98/conv2d_198/BiasAddBiasAdd7gcn_attention_49/core_gcn_98/conv2d_198/Conv2D:output:0Fgcn_attention_49/core_gcn_98/conv2d_198/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€21
/gcn_attention_49/core_gcn_98/conv2d_198/BiasAddк
,gcn_attention_49/core_gcn_98/activation/ReluRelu8gcn_attention_49/core_gcn_98/conv2d_198/BiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2.
,gcn_attention_49/core_gcn_98/activation/ReluУ
Cgcn_attention_49/core_gcn_98/batch_normalization_788/ReadVariableOpReadVariableOpLgcn_attention_49_core_gcn_98_batch_normalization_788_readvariableop_resource*
_output_shapes
:*
dtype02E
Cgcn_attention_49/core_gcn_98/batch_normalization_788/ReadVariableOpЩ
Egcn_attention_49/core_gcn_98/batch_normalization_788/ReadVariableOp_1ReadVariableOpNgcn_attention_49_core_gcn_98_batch_normalization_788_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Egcn_attention_49/core_gcn_98/batch_normalization_788/ReadVariableOp_1∆
Tgcn_attention_49/core_gcn_98/batch_normalization_788/FusedBatchNormV3/ReadVariableOpReadVariableOp]gcn_attention_49_core_gcn_98_batch_normalization_788_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02V
Tgcn_attention_49/core_gcn_98/batch_normalization_788/FusedBatchNormV3/ReadVariableOpћ
Vgcn_attention_49/core_gcn_98/batch_normalization_788/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_gcn_attention_49_core_gcn_98_batch_normalization_788_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02X
Vgcn_attention_49/core_gcn_98/batch_normalization_788/FusedBatchNormV3/ReadVariableOp_1ќ
Egcn_attention_49/core_gcn_98/batch_normalization_788/FusedBatchNormV3FusedBatchNormV3:gcn_attention_49/core_gcn_98/activation/Relu:activations:0Kgcn_attention_49/core_gcn_98/batch_normalization_788/ReadVariableOp:value:0Mgcn_attention_49/core_gcn_98/batch_normalization_788/ReadVariableOp_1:value:0\gcn_attention_49/core_gcn_98/batch_normalization_788/FusedBatchNormV3/ReadVariableOp:value:0^gcn_attention_49/core_gcn_98/batch_normalization_788/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2G
Egcn_attention_49/core_gcn_98/batch_normalization_788/FusedBatchNormV3Й
1gcn_attention_49/core_gcn_98/dropout_592/IdentityIdentityIgcn_attention_49/core_gcn_98/batch_normalization_788/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€23
1gcn_attention_49/core_gcn_98/dropout_592/Identity…
=gcn_attention_49/core_gcn_98/conv1d_393/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2?
=gcn_attention_49/core_gcn_98/conv1d_393/conv1d/ExpandDims/dimњ
9gcn_attention_49/core_gcn_98/conv1d_393/conv1d/ExpandDims
ExpandDims.gcn_attention_49/dropout_588/Identity:output:0Fgcn_attention_49/core_gcn_98/conv1d_393/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2;
9gcn_attention_49/core_gcn_98/conv1d_393/conv1d/ExpandDims∞
Jgcn_attention_49/core_gcn_98/conv1d_393/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_98_conv1d_393_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_98/conv1d_393/conv1d/ExpandDims_1/ReadVariableOpƒ
?gcn_attention_49/core_gcn_98/conv1d_393/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2A
?gcn_attention_49/core_gcn_98/conv1d_393/conv1d/ExpandDims_1/dim„
;gcn_attention_49/core_gcn_98/conv1d_393/conv1d/ExpandDims_1
ExpandDimsRgcn_attention_49/core_gcn_98/conv1d_393/conv1d/ExpandDims_1/ReadVariableOp:value:0Hgcn_attention_49/core_gcn_98/conv1d_393/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2=
;gcn_attention_49/core_gcn_98/conv1d_393/conv1d/ExpandDims_1я
.gcn_attention_49/core_gcn_98/conv1d_393/conv1dConv2DBgcn_attention_49/core_gcn_98/conv1d_393/conv1d/ExpandDims:output:0Dgcn_attention_49/core_gcn_98/conv1d_393/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
20
.gcn_attention_49/core_gcn_98/conv1d_393/conv1dУ
6gcn_attention_49/core_gcn_98/conv1d_393/conv1d/SqueezeSqueeze7gcn_attention_49/core_gcn_98/conv1d_393/conv1d:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
squeeze_dims

э€€€€€€€€28
6gcn_attention_49/core_gcn_98/conv1d_393/conv1d/SqueezeД
>gcn_attention_49/core_gcn_98/conv1d_393/BiasAdd/ReadVariableOpReadVariableOpGgcn_attention_49_core_gcn_98_conv1d_393_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_49/core_gcn_98/conv1d_393/BiasAdd/ReadVariableOpµ
/gcn_attention_49/core_gcn_98/conv1d_393/BiasAddBiasAdd?gcn_attention_49/core_gcn_98/conv1d_393/conv1d/Squeeze:output:0Fgcn_attention_49/core_gcn_98/conv1d_393/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€21
/gcn_attention_49/core_gcn_98/conv1d_393/BiasAddб
.gcn_attention_49/core_gcn_98/activation_1/ReluRelu8gcn_attention_49/core_gcn_98/conv1d_393/BiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€20
.gcn_attention_49/core_gcn_98/activation_1/ReluҐ
Hgcn_attention_49/core_gcn_98/batch_normalization_789/Cast/ReadVariableOpReadVariableOpQgcn_attention_49_core_gcn_98_batch_normalization_789_cast_readvariableop_resource*
_output_shapes
:*
dtype02J
Hgcn_attention_49/core_gcn_98/batch_normalization_789/Cast/ReadVariableOp®
Jgcn_attention_49/core_gcn_98/batch_normalization_789/Cast_1/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_98_batch_normalization_789_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_98/batch_normalization_789/Cast_1/ReadVariableOp®
Jgcn_attention_49/core_gcn_98/batch_normalization_789/Cast_2/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_98_batch_normalization_789_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_98/batch_normalization_789/Cast_2/ReadVariableOp®
Jgcn_attention_49/core_gcn_98/batch_normalization_789/Cast_3/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_98_batch_normalization_789_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_98/batch_normalization_789/Cast_3/ReadVariableOp—
Dgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2F
Dgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/add/yў
Bgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/addAddV2Rgcn_attention_49/core_gcn_98/batch_normalization_789/Cast_1/ReadVariableOp:value:0Mgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/add/y:output:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/addВ
Dgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/RsqrtRsqrtFgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/add:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/Rsqrt“
Bgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/mulMulHgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/Rsqrt:y:0Rgcn_attention_49/core_gcn_98/batch_normalization_789/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/mulЎ
Dgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/mul_1Mul<gcn_attention_49/core_gcn_98/activation_1/Relu:activations:0Fgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2F
Dgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/mul_1“
Dgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/mul_2MulPgcn_attention_49/core_gcn_98/batch_normalization_789/Cast/ReadVariableOp:value:0Fgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/mul:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/mul_2“
Bgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/subSubRgcn_attention_49/core_gcn_98/batch_normalization_789/Cast_2/ReadVariableOp:value:0Hgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/subж
Dgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/add_1AddV2Hgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/mul_1:z:0Fgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2F
Dgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/add_1ы
1gcn_attention_49/core_gcn_98/dropout_593/IdentityIdentityHgcn_attention_49/core_gcn_98/batch_normalization_789/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€23
1gcn_attention_49/core_gcn_98/dropout_593/IdentityЬ
+gcn_attention_49/core_gcn_98/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+gcn_attention_49/core_gcn_98/ExpandDims/dimХ
'gcn_attention_49/core_gcn_98/ExpandDims
ExpandDims:gcn_attention_49/core_gcn_98/dropout_593/Identity:output:04gcn_attention_49/core_gcn_98/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2)
'gcn_attention_49/core_gcn_98/ExpandDims†
-gcn_attention_49/core_gcn_98/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-gcn_attention_49/core_gcn_98/ExpandDims_1/dimЫ
)gcn_attention_49/core_gcn_98/ExpandDims_1
ExpandDims:gcn_attention_49/core_gcn_98/dropout_593/Identity:output:06gcn_attention_49/core_gcn_98/ExpandDims_1/dim:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2+
)gcn_attention_49/core_gcn_98/ExpandDims_1С
$gcn_attention_49/core_gcn_98/add/addAddV2:gcn_attention_49/core_gcn_98/dropout_592/Identity:output:02gcn_attention_49/core_gcn_98/ExpandDims_1:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2&
$gcn_attention_49/core_gcn_98/add/addБ
&gcn_attention_49/core_gcn_98/add/add_1AddV2(gcn_attention_49/core_gcn_98/add/add:z:00gcn_attention_49/core_gcn_98/ExpandDims:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2(
&gcn_attention_49/core_gcn_98/add/add_1й
1gcn_attention_49/core_gcn_98/activation_2/SigmoidSigmoid*gcn_attention_49/core_gcn_98/add/add_1:z:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€23
1gcn_attention_49/core_gcn_98/activation_2/Sigmoid…
=gcn_attention_49/core_gcn_98/conv1d_394/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2?
=gcn_attention_49/core_gcn_98/conv1d_394/conv1d/ExpandDims/dimњ
9gcn_attention_49/core_gcn_98/conv1d_394/conv1d/ExpandDims
ExpandDims.gcn_attention_49/dropout_588/Identity:output:0Fgcn_attention_49/core_gcn_98/conv1d_394/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2;
9gcn_attention_49/core_gcn_98/conv1d_394/conv1d/ExpandDims∞
Jgcn_attention_49/core_gcn_98/conv1d_394/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_98_conv1d_394_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_98/conv1d_394/conv1d/ExpandDims_1/ReadVariableOpƒ
?gcn_attention_49/core_gcn_98/conv1d_394/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2A
?gcn_attention_49/core_gcn_98/conv1d_394/conv1d/ExpandDims_1/dim„
;gcn_attention_49/core_gcn_98/conv1d_394/conv1d/ExpandDims_1
ExpandDimsRgcn_attention_49/core_gcn_98/conv1d_394/conv1d/ExpandDims_1/ReadVariableOp:value:0Hgcn_attention_49/core_gcn_98/conv1d_394/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2=
;gcn_attention_49/core_gcn_98/conv1d_394/conv1d/ExpandDims_1я
.gcn_attention_49/core_gcn_98/conv1d_394/conv1dConv2DBgcn_attention_49/core_gcn_98/conv1d_394/conv1d/ExpandDims:output:0Dgcn_attention_49/core_gcn_98/conv1d_394/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
20
.gcn_attention_49/core_gcn_98/conv1d_394/conv1dУ
6gcn_attention_49/core_gcn_98/conv1d_394/conv1d/SqueezeSqueeze7gcn_attention_49/core_gcn_98/conv1d_394/conv1d:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
squeeze_dims

э€€€€€€€€28
6gcn_attention_49/core_gcn_98/conv1d_394/conv1d/SqueezeД
>gcn_attention_49/core_gcn_98/conv1d_394/BiasAdd/ReadVariableOpReadVariableOpGgcn_attention_49_core_gcn_98_conv1d_394_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_49/core_gcn_98/conv1d_394/BiasAdd/ReadVariableOpµ
/gcn_attention_49/core_gcn_98/conv1d_394/BiasAddBiasAdd?gcn_attention_49/core_gcn_98/conv1d_394/conv1d/Squeeze:output:0Fgcn_attention_49/core_gcn_98/conv1d_394/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€21
/gcn_attention_49/core_gcn_98/conv1d_394/BiasAddҐ
Hgcn_attention_49/core_gcn_98/batch_normalization_790/Cast/ReadVariableOpReadVariableOpQgcn_attention_49_core_gcn_98_batch_normalization_790_cast_readvariableop_resource*
_output_shapes
:*
dtype02J
Hgcn_attention_49/core_gcn_98/batch_normalization_790/Cast/ReadVariableOp®
Jgcn_attention_49/core_gcn_98/batch_normalization_790/Cast_1/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_98_batch_normalization_790_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_98/batch_normalization_790/Cast_1/ReadVariableOp®
Jgcn_attention_49/core_gcn_98/batch_normalization_790/Cast_2/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_98_batch_normalization_790_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_98/batch_normalization_790/Cast_2/ReadVariableOp®
Jgcn_attention_49/core_gcn_98/batch_normalization_790/Cast_3/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_98_batch_normalization_790_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_98/batch_normalization_790/Cast_3/ReadVariableOp—
Dgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2F
Dgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/add/yў
Bgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/addAddV2Rgcn_attention_49/core_gcn_98/batch_normalization_790/Cast_1/ReadVariableOp:value:0Mgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/add/y:output:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/addВ
Dgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/RsqrtRsqrtFgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/add:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/Rsqrt“
Bgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/mulMulHgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/Rsqrt:y:0Rgcn_attention_49/core_gcn_98/batch_normalization_790/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/mul‘
Dgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/mul_1Mul8gcn_attention_49/core_gcn_98/conv1d_394/BiasAdd:output:0Fgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2F
Dgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/mul_1“
Dgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/mul_2MulPgcn_attention_49/core_gcn_98/batch_normalization_790/Cast/ReadVariableOp:value:0Fgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/mul:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/mul_2“
Bgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/subSubRgcn_attention_49/core_gcn_98/batch_normalization_790/Cast_2/ReadVariableOp:value:0Hgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/subж
Dgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/add_1AddV2Hgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/mul_1:z:0Fgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2F
Dgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/add_1с
.gcn_attention_49/core_gcn_98/activation_3/ReluReluHgcn_attention_49/core_gcn_98/batch_normalization_790/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€20
.gcn_attention_49/core_gcn_98/activation_3/Reluп
1gcn_attention_49/core_gcn_98/dropout_594/IdentityIdentity<gcn_attention_49/core_gcn_98/activation_3/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€23
1gcn_attention_49/core_gcn_98/dropout_594/Identity…
=gcn_attention_49/core_gcn_98/conv1d_395/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2?
=gcn_attention_49/core_gcn_98/conv1d_395/conv1d/ExpandDims/dimњ
9gcn_attention_49/core_gcn_98/conv1d_395/conv1d/ExpandDims
ExpandDims.gcn_attention_49/dropout_588/Identity:output:0Fgcn_attention_49/core_gcn_98/conv1d_395/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2;
9gcn_attention_49/core_gcn_98/conv1d_395/conv1d/ExpandDims∞
Jgcn_attention_49/core_gcn_98/conv1d_395/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_98_conv1d_395_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_98/conv1d_395/conv1d/ExpandDims_1/ReadVariableOpƒ
?gcn_attention_49/core_gcn_98/conv1d_395/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2A
?gcn_attention_49/core_gcn_98/conv1d_395/conv1d/ExpandDims_1/dim„
;gcn_attention_49/core_gcn_98/conv1d_395/conv1d/ExpandDims_1
ExpandDimsRgcn_attention_49/core_gcn_98/conv1d_395/conv1d/ExpandDims_1/ReadVariableOp:value:0Hgcn_attention_49/core_gcn_98/conv1d_395/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2=
;gcn_attention_49/core_gcn_98/conv1d_395/conv1d/ExpandDims_1я
.gcn_attention_49/core_gcn_98/conv1d_395/conv1dConv2DBgcn_attention_49/core_gcn_98/conv1d_395/conv1d/ExpandDims:output:0Dgcn_attention_49/core_gcn_98/conv1d_395/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
20
.gcn_attention_49/core_gcn_98/conv1d_395/conv1dУ
6gcn_attention_49/core_gcn_98/conv1d_395/conv1d/SqueezeSqueeze7gcn_attention_49/core_gcn_98/conv1d_395/conv1d:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
squeeze_dims

э€€€€€€€€28
6gcn_attention_49/core_gcn_98/conv1d_395/conv1d/SqueezeД
>gcn_attention_49/core_gcn_98/conv1d_395/BiasAdd/ReadVariableOpReadVariableOpGgcn_attention_49_core_gcn_98_conv1d_395_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_49/core_gcn_98/conv1d_395/BiasAdd/ReadVariableOpµ
/gcn_attention_49/core_gcn_98/conv1d_395/BiasAddBiasAdd?gcn_attention_49/core_gcn_98/conv1d_395/conv1d/Squeeze:output:0Fgcn_attention_49/core_gcn_98/conv1d_395/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€21
/gcn_attention_49/core_gcn_98/conv1d_395/BiasAddҐ
Hgcn_attention_49/core_gcn_98/batch_normalization_791/Cast/ReadVariableOpReadVariableOpQgcn_attention_49_core_gcn_98_batch_normalization_791_cast_readvariableop_resource*
_output_shapes
:*
dtype02J
Hgcn_attention_49/core_gcn_98/batch_normalization_791/Cast/ReadVariableOp®
Jgcn_attention_49/core_gcn_98/batch_normalization_791/Cast_1/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_98_batch_normalization_791_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_98/batch_normalization_791/Cast_1/ReadVariableOp®
Jgcn_attention_49/core_gcn_98/batch_normalization_791/Cast_2/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_98_batch_normalization_791_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_98/batch_normalization_791/Cast_2/ReadVariableOp®
Jgcn_attention_49/core_gcn_98/batch_normalization_791/Cast_3/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_98_batch_normalization_791_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_98/batch_normalization_791/Cast_3/ReadVariableOp—
Dgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2F
Dgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/add/yў
Bgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/addAddV2Rgcn_attention_49/core_gcn_98/batch_normalization_791/Cast_1/ReadVariableOp:value:0Mgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/add/y:output:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/addВ
Dgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/RsqrtRsqrtFgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/add:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/Rsqrt“
Bgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/mulMulHgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/Rsqrt:y:0Rgcn_attention_49/core_gcn_98/batch_normalization_791/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/mul‘
Dgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/mul_1Mul8gcn_attention_49/core_gcn_98/conv1d_395/BiasAdd:output:0Fgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2F
Dgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/mul_1“
Dgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/mul_2MulPgcn_attention_49/core_gcn_98/batch_normalization_791/Cast/ReadVariableOp:value:0Fgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/mul:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/mul_2“
Bgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/subSubRgcn_attention_49/core_gcn_98/batch_normalization_791/Cast_2/ReadVariableOp:value:0Hgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/subж
Dgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/add_1AddV2Hgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/mul_1:z:0Fgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2F
Dgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/add_1с
.gcn_attention_49/core_gcn_98/activation_4/ReluReluHgcn_attention_49/core_gcn_98/batch_normalization_791/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€20
.gcn_attention_49/core_gcn_98/activation_4/Reluп
1gcn_attention_49/core_gcn_98/dropout_595/IdentityIdentity<gcn_attention_49/core_gcn_98/activation_4/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€23
1gcn_attention_49/core_gcn_98/dropout_595/Identity†
-gcn_attention_49/core_gcn_98/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-gcn_attention_49/core_gcn_98/ExpandDims_2/dimЫ
)gcn_attention_49/core_gcn_98/ExpandDims_2
ExpandDims:gcn_attention_49/core_gcn_98/dropout_595/Identity:output:06gcn_attention_49/core_gcn_98/ExpandDims_2/dim:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2+
)gcn_attention_49/core_gcn_98/ExpandDims_2Ф
)gcn_attention_49/core_gcn_98/multiply/mulMul5gcn_attention_49/core_gcn_98/activation_2/Sigmoid:y:02gcn_attention_49/core_gcn_98/ExpandDims_2:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2+
)gcn_attention_49/core_gcn_98/multiply/mulЄ
9gcn_attention_49/core_gcn_98/lambda/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9gcn_attention_49/core_gcn_98/lambda/Sum/reduction_indicesЛ
'gcn_attention_49/core_gcn_98/lambda/SumSum-gcn_attention_49/core_gcn_98/multiply/mul:z:0Bgcn_attention_49/core_gcn_98/lambda/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2)
'gcn_attention_49/core_gcn_98/lambda/SumЉ
;gcn_attention_49/core_gcn_98/lambda/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;gcn_attention_49/core_gcn_98/lambda/Sum_1/reduction_indicesЩ
)gcn_attention_49/core_gcn_98/lambda/Sum_1Sum5gcn_attention_49/core_gcn_98/activation_2/Sigmoid:y:0Dgcn_attention_49/core_gcn_98/lambda/Sum_1/reduction_indices:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2+
)gcn_attention_49/core_gcn_98/lambda/Sum_1Я
+gcn_attention_49/core_gcn_98/lambda_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *е<2-
+gcn_attention_49/core_gcn_98/lambda_1/add/yИ
)gcn_attention_49/core_gcn_98/lambda_1/addAddV22gcn_attention_49/core_gcn_98/lambda/Sum_1:output:04gcn_attention_49/core_gcn_98/lambda_1/add/y:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2+
)gcn_attention_49/core_gcn_98/lambda_1/addЙ
-gcn_attention_49/core_gcn_98/lambda_1/truedivRealDiv0gcn_attention_49/core_gcn_98/lambda/Sum:output:0-gcn_attention_49/core_gcn_98/lambda_1/add:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2/
-gcn_attention_49/core_gcn_98/lambda_1/truedivЗ
&gcn_attention_49/core_gcn_98/add_1/addAddV2:gcn_attention_49/core_gcn_98/dropout_594/Identity:output:01gcn_attention_49/core_gcn_98/lambda_1/truediv:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2(
&gcn_attention_49/core_gcn_98/add_1/addУ
Cgcn_attention_49/core_gcn_98/batch_normalization_792/ReadVariableOpReadVariableOpLgcn_attention_49_core_gcn_98_batch_normalization_792_readvariableop_resource*
_output_shapes
:*
dtype02E
Cgcn_attention_49/core_gcn_98/batch_normalization_792/ReadVariableOpЩ
Egcn_attention_49/core_gcn_98/batch_normalization_792/ReadVariableOp_1ReadVariableOpNgcn_attention_49_core_gcn_98_batch_normalization_792_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Egcn_attention_49/core_gcn_98/batch_normalization_792/ReadVariableOp_1∆
Tgcn_attention_49/core_gcn_98/batch_normalization_792/FusedBatchNormV3/ReadVariableOpReadVariableOp]gcn_attention_49_core_gcn_98_batch_normalization_792_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02V
Tgcn_attention_49/core_gcn_98/batch_normalization_792/FusedBatchNormV3/ReadVariableOpћ
Vgcn_attention_49/core_gcn_98/batch_normalization_792/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_gcn_attention_49_core_gcn_98_batch_normalization_792_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02X
Vgcn_attention_49/core_gcn_98/batch_normalization_792/FusedBatchNormV3/ReadVariableOp_1Њ
Egcn_attention_49/core_gcn_98/batch_normalization_792/FusedBatchNormV3FusedBatchNormV3*gcn_attention_49/core_gcn_98/add/add_1:z:0Kgcn_attention_49/core_gcn_98/batch_normalization_792/ReadVariableOp:value:0Mgcn_attention_49/core_gcn_98/batch_normalization_792/ReadVariableOp_1:value:0\gcn_attention_49/core_gcn_98/batch_normalization_792/FusedBatchNormV3/ReadVariableOp:value:0^gcn_attention_49/core_gcn_98/batch_normalization_792/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2G
Egcn_attention_49/core_gcn_98/batch_normalization_792/FusedBatchNormV3Ґ
Hgcn_attention_49/core_gcn_98/batch_normalization_793/Cast/ReadVariableOpReadVariableOpQgcn_attention_49_core_gcn_98_batch_normalization_793_cast_readvariableop_resource*
_output_shapes
:*
dtype02J
Hgcn_attention_49/core_gcn_98/batch_normalization_793/Cast/ReadVariableOp®
Jgcn_attention_49/core_gcn_98/batch_normalization_793/Cast_1/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_98_batch_normalization_793_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_98/batch_normalization_793/Cast_1/ReadVariableOp®
Jgcn_attention_49/core_gcn_98/batch_normalization_793/Cast_2/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_98_batch_normalization_793_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_98/batch_normalization_793/Cast_2/ReadVariableOp®
Jgcn_attention_49/core_gcn_98/batch_normalization_793/Cast_3/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_98_batch_normalization_793_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_98/batch_normalization_793/Cast_3/ReadVariableOp—
Dgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2F
Dgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/add/yў
Bgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/addAddV2Rgcn_attention_49/core_gcn_98/batch_normalization_793/Cast_1/ReadVariableOp:value:0Mgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/add/y:output:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/addВ
Dgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/RsqrtRsqrtFgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/add:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/Rsqrt“
Bgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/mulMulHgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/Rsqrt:y:0Rgcn_attention_49/core_gcn_98/batch_normalization_793/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/mul∆
Dgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/mul_1Mul*gcn_attention_49/core_gcn_98/add_1/add:z:0Fgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2F
Dgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/mul_1“
Dgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/mul_2MulPgcn_attention_49/core_gcn_98/batch_normalization_793/Cast/ReadVariableOp:value:0Fgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/mul:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/mul_2“
Bgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/subSubRgcn_attention_49/core_gcn_98/batch_normalization_793/Cast_2/ReadVariableOp:value:0Hgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/subж
Dgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/add_1AddV2Hgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/mul_1:z:0Fgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2F
Dgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/add_1€
.gcn_attention_49/core_gcn_98/activation_5/ReluReluIgcn_attention_49/core_gcn_98/batch_normalization_792/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€20
.gcn_attention_49/core_gcn_98/activation_5/Reluс
.gcn_attention_49/core_gcn_98/activation_6/ReluReluHgcn_attention_49/core_gcn_98/batch_normalization_793/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€20
.gcn_attention_49/core_gcn_98/activation_6/ReluЖ
&gcn_attention_49/core_gcn_98/add_2/addAddV2.gcn_attention_49/dropout_588/Identity:output:0<gcn_attention_49/core_gcn_98/activation_6/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2(
&gcn_attention_49/core_gcn_98/add_2/addС
&gcn_attention_49/core_gcn_98/add_3/addAddV2,gcn_attention_49/concatenate/concat:output:0<gcn_attention_49/core_gcn_98/activation_5/Relu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2(
&gcn_attention_49/core_gcn_98/add_3/addЌ
<gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2>
<gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/dilation_rate”
;gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2=
;gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/filter_shapeЌ
4gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            26
4gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/stack∆
4gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/ShapeShape*gcn_attention_49/core_gcn_98/add_3/add:z:0*
T0*
_output_shapes
:26
4gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/Shape“
Bgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2D
Bgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice/stack÷
Dgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice/stack_1÷
Dgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice/stack_2ь
<gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_sliceStridedSlice=gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/Shape:output:0Kgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice/stack:output:0Mgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice/stack_1:output:0Mgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice÷
Dgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2F
Dgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_1/stackЏ
Fgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_1/stack_1Џ
Fgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_1/stack_2Ж
>gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_1StridedSlice=gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/Shape:output:0Mgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_1/stack:output:0Ogcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_1/stack_1:output:0Ogcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2@
>gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_1ґ
6gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/stack_1PackEgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice:output:0Ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_1:output:0*
N*
T0*
_output_shapes
:28
6gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/stack_1Ы
cgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2e
cgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice/stackЯ
egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2g
egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice/stack_1Я
egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2g
egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice/stack_2«
]gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_sliceStridedSlice=gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/stack:output:0lgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice/stack:output:0ngcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice/stack_1:output:0ngcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2_
]gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_sliceЯ
egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2g
egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_1/stack£
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2i
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_1/stack_1£
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2i
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_1/stack_2—
_gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_1StridedSlice=gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/stack:output:0ngcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_1/stack:output:0pgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_1/stack_1:output:0pgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2a
_gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_1Б
Sgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/addAddV2?gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/stack_1:output:0fgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice:output:0*
T0*
_output_shapes
:2U
Sgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/addЯ
Ugcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/add_1AddV2Wgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/add:z:0hgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_1:output:0*
T0*
_output_shapes
:2W
Ugcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/add_1э
Sgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/modFloorModYgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/add_1:z:0Egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/dilation_rate:output:0*
T0*
_output_shapes
:2U
Sgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/modц
Sgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/subSubEgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/dilation_rate:output:0Wgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/mod:z:0*
T0*
_output_shapes
:2U
Sgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/sub€
Ugcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/mod_1FloorModWgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/sub:z:0Egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/dilation_rate:output:0*
T0*
_output_shapes
:2W
Ugcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/mod_1°
Ugcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/add_2AddV2hgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_1:output:0Ygcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/mod_1:z:0*
T0*
_output_shapes
:2W
Ugcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/add_2Ш
egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_2/stackЬ
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_2/stack_1Ь
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_2/stack_2‘
_gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_2StridedSlicefgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice:output:0ngcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_2/stack:output:0pgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_2/stack_1:output:0pgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_2Ш
egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_3/stackЬ
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_3/stack_1Ь
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_3/stack_2«
_gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_3StridedSliceYgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/add_2:z:0ngcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_3/stack:output:0pgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_3/stack_1:output:0pgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_3Ш
egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2g
egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_4/stackЬ
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_4/stack_1Ь
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_4/stack_2‘
_gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_4StridedSlicefgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice:output:0ngcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_4/stack:output:0pgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_4/stack_1:output:0pgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_4Ш
egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2g
egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_5/stackЬ
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_5/stack_1Ь
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_5/stack_2«
_gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_5StridedSliceYgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/add_2:z:0ngcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_5/stack:output:0pgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_5/stack_1:output:0pgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_5¬
Zgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/paddings/0Packhgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_2:output:0hgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2\
Zgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/paddings/0¬
Zgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/paddings/1Packhgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_4:output:0hgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_5:output:0*
N*
T0*
_output_shapes
:2\
Zgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/paddings/1Є
Xgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/paddingsPackcgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/paddings/0:output:0cgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/paddings/1:output:0*
N*
T0*
_output_shapes

:2Z
Xgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/paddingsШ
egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_6/stackЬ
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_6/stack_1Ь
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_6/stack_2«
_gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_6StridedSliceYgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/mod_1:z:0ngcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_6/stack:output:0pgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_6/stack_1:output:0pgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_6Ш
egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2g
egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_7/stackЬ
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_7/stack_1Ь
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_7/stack_2«
_gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_7StridedSliceYgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/mod_1:z:0ngcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_7/stack:output:0pgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_7/stack_1:output:0pgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_7ш
Ygcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/crops/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2[
Ygcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/crops/0/0ґ
Wgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/crops/0Packbgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/crops/0/0:output:0hgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_6:output:0*
N*
T0*
_output_shapes
:2Y
Wgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/crops/0ш
Ygcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/crops/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2[
Ygcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/crops/1/0ґ
Wgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/crops/1Packbgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/crops/1/0:output:0hgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/strided_slice_7:output:0*
N*
T0*
_output_shapes
:2Y
Wgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/crops/1ђ
Ugcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/cropsPack`gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/crops/0:output:0`gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/crops/1:output:0*
N*
T0*
_output_shapes

:2W
Ugcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/crops÷
Dgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_2/stackЏ
Fgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_2/stack_1Џ
Fgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_2/stack_2Ъ
>gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_2StridedSliceagcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/paddings:output:0Mgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_2/stack:output:0Ogcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_2/stack_1:output:0Ogcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:2@
>gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_2∆
@gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 2B
@gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/concat/concat_dimъ
<gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/concat/concatIdentityGgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_2:output:0*
T0*
_output_shapes

:2>
<gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/concat/concat÷
Dgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_3/stackЏ
Fgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_3/stack_1Џ
Fgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_3/stack_2Ч
>gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_3StridedSlice^gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/required_space_to_batch_paddings/crops:output:0Mgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_3/stack:output:0Ogcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_3/stack_1:output:0Ogcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:2@
>gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_3 
Bgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/concat_1/concat_dimю
>gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/concat_1/concatIdentityGgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/strided_slice_3:output:0*
T0*
_output_shapes

:2@
>gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/concat_1/concatз
Igcn_attention_49/core_gcn_99/conv2d_199/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2K
Igcn_attention_49/core_gcn_99/conv2d_199/Conv2D/SpaceToBatchND/block_shape£
=gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/SpaceToBatchNDSpaceToBatchND*gcn_attention_49/core_gcn_98/add_3/add:z:0Rgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/SpaceToBatchND/block_shape:output:0Egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/concat/concat:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2?
=gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/SpaceToBatchNDН
=gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/ReadVariableOpReadVariableOpFgcn_attention_49_core_gcn_99_conv2d_199_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02?
=gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/ReadVariableOpо
.gcn_attention_49/core_gcn_99/conv2d_199/Conv2DConv2DFgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/SpaceToBatchND:output:0Egcn_attention_49/core_gcn_99/conv2d_199/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingVALID*
strides
20
.gcn_attention_49/core_gcn_99/conv2d_199/Conv2Dз
Igcn_attention_49/core_gcn_99/conv2d_199/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2K
Igcn_attention_49/core_gcn_99/conv2d_199/Conv2D/BatchToSpaceND/block_shape≤
=gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/BatchToSpaceNDBatchToSpaceND7gcn_attention_49/core_gcn_99/conv2d_199/Conv2D:output:0Rgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/BatchToSpaceND/block_shape:output:0Ggcn_attention_49/core_gcn_99/conv2d_199/Conv2D/concat_1/concat:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2?
=gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/BatchToSpaceNDД
>gcn_attention_49/core_gcn_99/conv2d_199/BiasAdd/ReadVariableOpReadVariableOpGgcn_attention_49_core_gcn_99_conv2d_199_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_49/core_gcn_99/conv2d_199/BiasAdd/ReadVariableOp…
/gcn_attention_49/core_gcn_99/conv2d_199/BiasAddBiasAddFgcn_attention_49/core_gcn_99/conv2d_199/Conv2D/BatchToSpaceND:output:0Fgcn_attention_49/core_gcn_99/conv2d_199/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€21
/gcn_attention_49/core_gcn_99/conv2d_199/BiasAddо
.gcn_attention_49/core_gcn_99/activation_7/ReluRelu8gcn_attention_49/core_gcn_99/conv2d_199/BiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€20
.gcn_attention_49/core_gcn_99/activation_7/ReluУ
Cgcn_attention_49/core_gcn_99/batch_normalization_794/ReadVariableOpReadVariableOpLgcn_attention_49_core_gcn_99_batch_normalization_794_readvariableop_resource*
_output_shapes
:*
dtype02E
Cgcn_attention_49/core_gcn_99/batch_normalization_794/ReadVariableOpЩ
Egcn_attention_49/core_gcn_99/batch_normalization_794/ReadVariableOp_1ReadVariableOpNgcn_attention_49_core_gcn_99_batch_normalization_794_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Egcn_attention_49/core_gcn_99/batch_normalization_794/ReadVariableOp_1∆
Tgcn_attention_49/core_gcn_99/batch_normalization_794/FusedBatchNormV3/ReadVariableOpReadVariableOp]gcn_attention_49_core_gcn_99_batch_normalization_794_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02V
Tgcn_attention_49/core_gcn_99/batch_normalization_794/FusedBatchNormV3/ReadVariableOpћ
Vgcn_attention_49/core_gcn_99/batch_normalization_794/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_gcn_attention_49_core_gcn_99_batch_normalization_794_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02X
Vgcn_attention_49/core_gcn_99/batch_normalization_794/FusedBatchNormV3/ReadVariableOp_1–
Egcn_attention_49/core_gcn_99/batch_normalization_794/FusedBatchNormV3FusedBatchNormV3<gcn_attention_49/core_gcn_99/activation_7/Relu:activations:0Kgcn_attention_49/core_gcn_99/batch_normalization_794/ReadVariableOp:value:0Mgcn_attention_49/core_gcn_99/batch_normalization_794/ReadVariableOp_1:value:0\gcn_attention_49/core_gcn_99/batch_normalization_794/FusedBatchNormV3/ReadVariableOp:value:0^gcn_attention_49/core_gcn_99/batch_normalization_794/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2G
Egcn_attention_49/core_gcn_99/batch_normalization_794/FusedBatchNormV3Й
1gcn_attention_49/core_gcn_99/dropout_596/IdentityIdentityIgcn_attention_49/core_gcn_99/batch_normalization_794/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€23
1gcn_attention_49/core_gcn_99/dropout_596/Identity∆
<gcn_attention_49/core_gcn_99/conv1d_396/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:2>
<gcn_attention_49/core_gcn_99/conv1d_396/conv1d/dilation_rateѕ
;gcn_attention_49/core_gcn_99/conv1d_396/conv1d/filter_shapeConst*
_output_shapes
:*
dtype0*!
valueB"         2=
;gcn_attention_49/core_gcn_99/conv1d_396/conv1d/filter_shape≈
4gcn_attention_49/core_gcn_99/conv1d_396/conv1d/stackConst*
_output_shapes

:*
dtype0*!
valueB"      26
4gcn_attention_49/core_gcn_99/conv1d_396/conv1d/stack∆
4gcn_attention_49/core_gcn_99/conv1d_396/conv1d/ShapeShape*gcn_attention_49/core_gcn_98/add_2/add:z:0*
T0*
_output_shapes
:26
4gcn_attention_49/core_gcn_99/conv1d_396/conv1d/Shape“
Bgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2D
Bgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice/stack÷
Dgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice/stack_1÷
Dgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice/stack_2ь
<gcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_sliceStridedSlice=gcn_attention_49/core_gcn_99/conv1d_396/conv1d/Shape:output:0Kgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice/stack:output:0Mgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice/stack_1:output:0Mgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<gcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_sliceн
6gcn_attention_49/core_gcn_99/conv1d_396/conv1d/stack_1PackEgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice:output:0*
N*
T0*
_output_shapes
:28
6gcn_attention_49/core_gcn_99/conv1d_396/conv1d/stack_1Ы
cgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2e
cgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice/stackЯ
egcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2g
egcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice/stack_1Я
egcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2g
egcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice/stack_2«
]gcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_sliceStridedSlice=gcn_attention_49/core_gcn_99/conv1d_396/conv1d/stack:output:0lgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice/stack:output:0ngcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice/stack_1:output:0ngcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2_
]gcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_sliceЯ
egcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2g
egcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_1/stack£
ggcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2i
ggcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1£
ggcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2i
ggcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2—
_gcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_1StridedSlice=gcn_attention_49/core_gcn_99/conv1d_396/conv1d/stack:output:0ngcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_1/stack:output:0pgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1:output:0pgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2a
_gcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_1Б
Sgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/addAddV2?gcn_attention_49/core_gcn_99/conv1d_396/conv1d/stack_1:output:0fgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice:output:0*
T0*
_output_shapes
:2U
Sgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/addЯ
Ugcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/add_1AddV2Wgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/add:z:0hgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_1:output:0*
T0*
_output_shapes
:2W
Ugcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/add_1э
Sgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/modFloorModYgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/add_1:z:0Egcn_attention_49/core_gcn_99/conv1d_396/conv1d/dilation_rate:output:0*
T0*
_output_shapes
:2U
Sgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/modц
Sgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/subSubEgcn_attention_49/core_gcn_99/conv1d_396/conv1d/dilation_rate:output:0Wgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/mod:z:0*
T0*
_output_shapes
:2U
Sgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/sub€
Ugcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/mod_1FloorModWgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/sub:z:0Egcn_attention_49/core_gcn_99/conv1d_396/conv1d/dilation_rate:output:0*
T0*
_output_shapes
:2W
Ugcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/mod_1°
Ugcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/add_2AddV2hgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_1:output:0Ygcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/mod_1:z:0*
T0*
_output_shapes
:2W
Ugcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/add_2Ш
egcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_2/stackЬ
ggcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1Ь
ggcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2‘
_gcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_2StridedSlicefgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice:output:0ngcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_2/stack:output:0pgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1:output:0pgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_2Ш
egcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_3/stackЬ
ggcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1Ь
ggcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2«
_gcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_3StridedSliceYgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/add_2:z:0ngcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_3/stack:output:0pgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1:output:0pgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_3¬
Zgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/paddings/0Packhgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_2:output:0hgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2\
Zgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/paddings/0”
Xgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/paddingsPackcgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/paddings/0:output:0*
N*
T0*
_output_shapes

:2Z
Xgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/paddingsШ
egcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_4/stackЬ
ggcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1Ь
ggcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2«
_gcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_4StridedSliceYgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/mod_1:z:0ngcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_4/stack:output:0pgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1:output:0pgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_4ш
Ygcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/crops/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2[
Ygcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/crops/0/0ґ
Wgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/crops/0Packbgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/crops/0/0:output:0hgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/strided_slice_4:output:0*
N*
T0*
_output_shapes
:2Y
Wgcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/crops/0 
Ugcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/cropsPack`gcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/crops/0:output:0*
N*
T0*
_output_shapes

:2W
Ugcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/crops÷
Dgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_1/stackЏ
Fgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_1/stack_1Џ
Fgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_1/stack_2Ъ
>gcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_1StridedSliceagcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/paddings:output:0Mgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_1/stack:output:0Ogcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_1/stack_1:output:0Ogcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:2@
>gcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_1∆
@gcn_attention_49/core_gcn_99/conv1d_396/conv1d/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 2B
@gcn_attention_49/core_gcn_99/conv1d_396/conv1d/concat/concat_dimъ
<gcn_attention_49/core_gcn_99/conv1d_396/conv1d/concat/concatIdentityGgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_1:output:0*
T0*
_output_shapes

:2>
<gcn_attention_49/core_gcn_99/conv1d_396/conv1d/concat/concat÷
Dgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_2/stackЏ
Fgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_2/stack_1Џ
Fgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_2/stack_2Ч
>gcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_2StridedSlice^gcn_attention_49/core_gcn_99/conv1d_396/conv1d/required_space_to_batch_paddings/crops:output:0Mgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_2/stack:output:0Ogcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_2/stack_1:output:0Ogcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:2@
>gcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_2 
Bgcn_attention_49/core_gcn_99/conv1d_396/conv1d/concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bgcn_attention_49/core_gcn_99/conv1d_396/conv1d/concat_1/concat_dimю
>gcn_attention_49/core_gcn_99/conv1d_396/conv1d/concat_1/concatIdentityGgcn_attention_49/core_gcn_99/conv1d_396/conv1d/strided_slice_2:output:0*
T0*
_output_shapes

:2@
>gcn_attention_49/core_gcn_99/conv1d_396/conv1d/concat_1/concatа
Igcn_attention_49/core_gcn_99/conv1d_396/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2K
Igcn_attention_49/core_gcn_99/conv1d_396/conv1d/SpaceToBatchND/block_shapeЦ
=gcn_attention_49/core_gcn_99/conv1d_396/conv1d/SpaceToBatchNDSpaceToBatchND*gcn_attention_49/core_gcn_98/add_2/add:z:0Rgcn_attention_49/core_gcn_99/conv1d_396/conv1d/SpaceToBatchND/block_shape:output:0Egcn_attention_49/core_gcn_99/conv1d_396/conv1d/concat/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2?
=gcn_attention_49/core_gcn_99/conv1d_396/conv1d/SpaceToBatchND…
=gcn_attention_49/core_gcn_99/conv1d_396/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2?
=gcn_attention_49/core_gcn_99/conv1d_396/conv1d/ExpandDims/dim„
9gcn_attention_49/core_gcn_99/conv1d_396/conv1d/ExpandDims
ExpandDimsFgcn_attention_49/core_gcn_99/conv1d_396/conv1d/SpaceToBatchND:output:0Fgcn_attention_49/core_gcn_99/conv1d_396/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2;
9gcn_attention_49/core_gcn_99/conv1d_396/conv1d/ExpandDims∞
Jgcn_attention_49/core_gcn_99/conv1d_396/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_99_conv1d_396_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_99/conv1d_396/conv1d/ExpandDims_1/ReadVariableOpƒ
?gcn_attention_49/core_gcn_99/conv1d_396/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2A
?gcn_attention_49/core_gcn_99/conv1d_396/conv1d/ExpandDims_1/dim„
;gcn_attention_49/core_gcn_99/conv1d_396/conv1d/ExpandDims_1
ExpandDimsRgcn_attention_49/core_gcn_99/conv1d_396/conv1d/ExpandDims_1/ReadVariableOp:value:0Hgcn_attention_49/core_gcn_99/conv1d_396/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2=
;gcn_attention_49/core_gcn_99/conv1d_396/conv1d/ExpandDims_1а
.gcn_attention_49/core_gcn_99/conv1d_396/conv1dConv2DBgcn_attention_49/core_gcn_99/conv1d_396/conv1d/ExpandDims:output:0Dgcn_attention_49/core_gcn_99/conv1d_396/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
paddingVALID*
strides
20
.gcn_attention_49/core_gcn_99/conv1d_396/conv1dУ
6gcn_attention_49/core_gcn_99/conv1d_396/conv1d/SqueezeSqueeze7gcn_attention_49/core_gcn_99/conv1d_396/conv1d:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
squeeze_dims

э€€€€€€€€28
6gcn_attention_49/core_gcn_99/conv1d_396/conv1d/Squeezeа
Igcn_attention_49/core_gcn_99/conv1d_396/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2K
Igcn_attention_49/core_gcn_99/conv1d_396/conv1d/BatchToSpaceND/block_shape≠
=gcn_attention_49/core_gcn_99/conv1d_396/conv1d/BatchToSpaceNDBatchToSpaceND?gcn_attention_49/core_gcn_99/conv1d_396/conv1d/Squeeze:output:0Rgcn_attention_49/core_gcn_99/conv1d_396/conv1d/BatchToSpaceND/block_shape:output:0Ggcn_attention_49/core_gcn_99/conv1d_396/conv1d/concat_1/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2?
=gcn_attention_49/core_gcn_99/conv1d_396/conv1d/BatchToSpaceNDД
>gcn_attention_49/core_gcn_99/conv1d_396/BiasAdd/ReadVariableOpReadVariableOpGgcn_attention_49_core_gcn_99_conv1d_396_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_49/core_gcn_99/conv1d_396/BiasAdd/ReadVariableOpЉ
/gcn_attention_49/core_gcn_99/conv1d_396/BiasAddBiasAddFgcn_attention_49/core_gcn_99/conv1d_396/conv1d/BatchToSpaceND:output:0Fgcn_attention_49/core_gcn_99/conv1d_396/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€21
/gcn_attention_49/core_gcn_99/conv1d_396/BiasAddб
.gcn_attention_49/core_gcn_99/activation_8/ReluRelu8gcn_attention_49/core_gcn_99/conv1d_396/BiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€20
.gcn_attention_49/core_gcn_99/activation_8/ReluҐ
Hgcn_attention_49/core_gcn_99/batch_normalization_795/Cast/ReadVariableOpReadVariableOpQgcn_attention_49_core_gcn_99_batch_normalization_795_cast_readvariableop_resource*
_output_shapes
:*
dtype02J
Hgcn_attention_49/core_gcn_99/batch_normalization_795/Cast/ReadVariableOp®
Jgcn_attention_49/core_gcn_99/batch_normalization_795/Cast_1/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_99_batch_normalization_795_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_99/batch_normalization_795/Cast_1/ReadVariableOp®
Jgcn_attention_49/core_gcn_99/batch_normalization_795/Cast_2/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_99_batch_normalization_795_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_99/batch_normalization_795/Cast_2/ReadVariableOp®
Jgcn_attention_49/core_gcn_99/batch_normalization_795/Cast_3/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_99_batch_normalization_795_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_99/batch_normalization_795/Cast_3/ReadVariableOp—
Dgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2F
Dgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/add/yў
Bgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/addAddV2Rgcn_attention_49/core_gcn_99/batch_normalization_795/Cast_1/ReadVariableOp:value:0Mgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/add/y:output:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/addВ
Dgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/RsqrtRsqrtFgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/add:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/Rsqrt“
Bgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/mulMulHgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/Rsqrt:y:0Rgcn_attention_49/core_gcn_99/batch_normalization_795/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/mulЎ
Dgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/mul_1Mul<gcn_attention_49/core_gcn_99/activation_8/Relu:activations:0Fgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2F
Dgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/mul_1“
Dgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/mul_2MulPgcn_attention_49/core_gcn_99/batch_normalization_795/Cast/ReadVariableOp:value:0Fgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/mul:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/mul_2“
Bgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/subSubRgcn_attention_49/core_gcn_99/batch_normalization_795/Cast_2/ReadVariableOp:value:0Hgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/subж
Dgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/add_1AddV2Hgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/mul_1:z:0Fgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2F
Dgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/add_1ы
1gcn_attention_49/core_gcn_99/dropout_597/IdentityIdentityHgcn_attention_49/core_gcn_99/batch_normalization_795/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€23
1gcn_attention_49/core_gcn_99/dropout_597/IdentityЬ
+gcn_attention_49/core_gcn_99/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+gcn_attention_49/core_gcn_99/ExpandDims/dimХ
'gcn_attention_49/core_gcn_99/ExpandDims
ExpandDims:gcn_attention_49/core_gcn_99/dropout_597/Identity:output:04gcn_attention_49/core_gcn_99/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2)
'gcn_attention_49/core_gcn_99/ExpandDims†
-gcn_attention_49/core_gcn_99/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-gcn_attention_49/core_gcn_99/ExpandDims_1/dimЫ
)gcn_attention_49/core_gcn_99/ExpandDims_1
ExpandDims:gcn_attention_49/core_gcn_99/dropout_597/Identity:output:06gcn_attention_49/core_gcn_99/ExpandDims_1/dim:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2+
)gcn_attention_49/core_gcn_99/ExpandDims_1Х
&gcn_attention_49/core_gcn_99/add_4/addAddV2:gcn_attention_49/core_gcn_99/dropout_596/Identity:output:02gcn_attention_49/core_gcn_99/ExpandDims_1:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2(
&gcn_attention_49/core_gcn_99/add_4/addЗ
(gcn_attention_49/core_gcn_99/add_4/add_1AddV2*gcn_attention_49/core_gcn_99/add_4/add:z:00gcn_attention_49/core_gcn_99/ExpandDims:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2*
(gcn_attention_49/core_gcn_99/add_4/add_1л
1gcn_attention_49/core_gcn_99/activation_9/SigmoidSigmoid,gcn_attention_49/core_gcn_99/add_4/add_1:z:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€23
1gcn_attention_49/core_gcn_99/activation_9/Sigmoid∆
<gcn_attention_49/core_gcn_99/conv1d_397/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:2>
<gcn_attention_49/core_gcn_99/conv1d_397/conv1d/dilation_rateѕ
;gcn_attention_49/core_gcn_99/conv1d_397/conv1d/filter_shapeConst*
_output_shapes
:*
dtype0*!
valueB"         2=
;gcn_attention_49/core_gcn_99/conv1d_397/conv1d/filter_shape≈
4gcn_attention_49/core_gcn_99/conv1d_397/conv1d/stackConst*
_output_shapes

:*
dtype0*!
valueB"      26
4gcn_attention_49/core_gcn_99/conv1d_397/conv1d/stack∆
4gcn_attention_49/core_gcn_99/conv1d_397/conv1d/ShapeShape*gcn_attention_49/core_gcn_98/add_2/add:z:0*
T0*
_output_shapes
:26
4gcn_attention_49/core_gcn_99/conv1d_397/conv1d/Shape“
Bgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2D
Bgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice/stack÷
Dgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice/stack_1÷
Dgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice/stack_2ь
<gcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_sliceStridedSlice=gcn_attention_49/core_gcn_99/conv1d_397/conv1d/Shape:output:0Kgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice/stack:output:0Mgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice/stack_1:output:0Mgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<gcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_sliceн
6gcn_attention_49/core_gcn_99/conv1d_397/conv1d/stack_1PackEgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice:output:0*
N*
T0*
_output_shapes
:28
6gcn_attention_49/core_gcn_99/conv1d_397/conv1d/stack_1Ы
cgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2e
cgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice/stackЯ
egcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2g
egcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice/stack_1Я
egcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2g
egcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice/stack_2«
]gcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_sliceStridedSlice=gcn_attention_49/core_gcn_99/conv1d_397/conv1d/stack:output:0lgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice/stack:output:0ngcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice/stack_1:output:0ngcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2_
]gcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_sliceЯ
egcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2g
egcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_1/stack£
ggcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2i
ggcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1£
ggcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2i
ggcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2—
_gcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_1StridedSlice=gcn_attention_49/core_gcn_99/conv1d_397/conv1d/stack:output:0ngcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_1/stack:output:0pgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1:output:0pgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2a
_gcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_1Б
Sgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/addAddV2?gcn_attention_49/core_gcn_99/conv1d_397/conv1d/stack_1:output:0fgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice:output:0*
T0*
_output_shapes
:2U
Sgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/addЯ
Ugcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/add_1AddV2Wgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/add:z:0hgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_1:output:0*
T0*
_output_shapes
:2W
Ugcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/add_1э
Sgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/modFloorModYgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/add_1:z:0Egcn_attention_49/core_gcn_99/conv1d_397/conv1d/dilation_rate:output:0*
T0*
_output_shapes
:2U
Sgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/modц
Sgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/subSubEgcn_attention_49/core_gcn_99/conv1d_397/conv1d/dilation_rate:output:0Wgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/mod:z:0*
T0*
_output_shapes
:2U
Sgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/sub€
Ugcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/mod_1FloorModWgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/sub:z:0Egcn_attention_49/core_gcn_99/conv1d_397/conv1d/dilation_rate:output:0*
T0*
_output_shapes
:2W
Ugcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/mod_1°
Ugcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/add_2AddV2hgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_1:output:0Ygcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/mod_1:z:0*
T0*
_output_shapes
:2W
Ugcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/add_2Ш
egcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_2/stackЬ
ggcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1Ь
ggcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2‘
_gcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_2StridedSlicefgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice:output:0ngcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_2/stack:output:0pgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1:output:0pgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_2Ш
egcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_3/stackЬ
ggcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1Ь
ggcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2«
_gcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_3StridedSliceYgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/add_2:z:0ngcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_3/stack:output:0pgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1:output:0pgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_3¬
Zgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/paddings/0Packhgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_2:output:0hgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2\
Zgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/paddings/0”
Xgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/paddingsPackcgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/paddings/0:output:0*
N*
T0*
_output_shapes

:2Z
Xgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/paddingsШ
egcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_4/stackЬ
ggcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1Ь
ggcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2«
_gcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_4StridedSliceYgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/mod_1:z:0ngcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_4/stack:output:0pgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1:output:0pgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_4ш
Ygcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/crops/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2[
Ygcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/crops/0/0ґ
Wgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/crops/0Packbgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/crops/0/0:output:0hgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/strided_slice_4:output:0*
N*
T0*
_output_shapes
:2Y
Wgcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/crops/0 
Ugcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/cropsPack`gcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/crops/0:output:0*
N*
T0*
_output_shapes

:2W
Ugcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/crops÷
Dgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_1/stackЏ
Fgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_1/stack_1Џ
Fgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_1/stack_2Ъ
>gcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_1StridedSliceagcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/paddings:output:0Mgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_1/stack:output:0Ogcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_1/stack_1:output:0Ogcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:2@
>gcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_1∆
@gcn_attention_49/core_gcn_99/conv1d_397/conv1d/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 2B
@gcn_attention_49/core_gcn_99/conv1d_397/conv1d/concat/concat_dimъ
<gcn_attention_49/core_gcn_99/conv1d_397/conv1d/concat/concatIdentityGgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_1:output:0*
T0*
_output_shapes

:2>
<gcn_attention_49/core_gcn_99/conv1d_397/conv1d/concat/concat÷
Dgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_2/stackЏ
Fgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_2/stack_1Џ
Fgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_2/stack_2Ч
>gcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_2StridedSlice^gcn_attention_49/core_gcn_99/conv1d_397/conv1d/required_space_to_batch_paddings/crops:output:0Mgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_2/stack:output:0Ogcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_2/stack_1:output:0Ogcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:2@
>gcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_2 
Bgcn_attention_49/core_gcn_99/conv1d_397/conv1d/concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bgcn_attention_49/core_gcn_99/conv1d_397/conv1d/concat_1/concat_dimю
>gcn_attention_49/core_gcn_99/conv1d_397/conv1d/concat_1/concatIdentityGgcn_attention_49/core_gcn_99/conv1d_397/conv1d/strided_slice_2:output:0*
T0*
_output_shapes

:2@
>gcn_attention_49/core_gcn_99/conv1d_397/conv1d/concat_1/concatа
Igcn_attention_49/core_gcn_99/conv1d_397/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2K
Igcn_attention_49/core_gcn_99/conv1d_397/conv1d/SpaceToBatchND/block_shapeЦ
=gcn_attention_49/core_gcn_99/conv1d_397/conv1d/SpaceToBatchNDSpaceToBatchND*gcn_attention_49/core_gcn_98/add_2/add:z:0Rgcn_attention_49/core_gcn_99/conv1d_397/conv1d/SpaceToBatchND/block_shape:output:0Egcn_attention_49/core_gcn_99/conv1d_397/conv1d/concat/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2?
=gcn_attention_49/core_gcn_99/conv1d_397/conv1d/SpaceToBatchND…
=gcn_attention_49/core_gcn_99/conv1d_397/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2?
=gcn_attention_49/core_gcn_99/conv1d_397/conv1d/ExpandDims/dim„
9gcn_attention_49/core_gcn_99/conv1d_397/conv1d/ExpandDims
ExpandDimsFgcn_attention_49/core_gcn_99/conv1d_397/conv1d/SpaceToBatchND:output:0Fgcn_attention_49/core_gcn_99/conv1d_397/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2;
9gcn_attention_49/core_gcn_99/conv1d_397/conv1d/ExpandDims∞
Jgcn_attention_49/core_gcn_99/conv1d_397/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_99_conv1d_397_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_99/conv1d_397/conv1d/ExpandDims_1/ReadVariableOpƒ
?gcn_attention_49/core_gcn_99/conv1d_397/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2A
?gcn_attention_49/core_gcn_99/conv1d_397/conv1d/ExpandDims_1/dim„
;gcn_attention_49/core_gcn_99/conv1d_397/conv1d/ExpandDims_1
ExpandDimsRgcn_attention_49/core_gcn_99/conv1d_397/conv1d/ExpandDims_1/ReadVariableOp:value:0Hgcn_attention_49/core_gcn_99/conv1d_397/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2=
;gcn_attention_49/core_gcn_99/conv1d_397/conv1d/ExpandDims_1а
.gcn_attention_49/core_gcn_99/conv1d_397/conv1dConv2DBgcn_attention_49/core_gcn_99/conv1d_397/conv1d/ExpandDims:output:0Dgcn_attention_49/core_gcn_99/conv1d_397/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
paddingVALID*
strides
20
.gcn_attention_49/core_gcn_99/conv1d_397/conv1dУ
6gcn_attention_49/core_gcn_99/conv1d_397/conv1d/SqueezeSqueeze7gcn_attention_49/core_gcn_99/conv1d_397/conv1d:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
squeeze_dims

э€€€€€€€€28
6gcn_attention_49/core_gcn_99/conv1d_397/conv1d/Squeezeа
Igcn_attention_49/core_gcn_99/conv1d_397/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2K
Igcn_attention_49/core_gcn_99/conv1d_397/conv1d/BatchToSpaceND/block_shape≠
=gcn_attention_49/core_gcn_99/conv1d_397/conv1d/BatchToSpaceNDBatchToSpaceND?gcn_attention_49/core_gcn_99/conv1d_397/conv1d/Squeeze:output:0Rgcn_attention_49/core_gcn_99/conv1d_397/conv1d/BatchToSpaceND/block_shape:output:0Ggcn_attention_49/core_gcn_99/conv1d_397/conv1d/concat_1/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2?
=gcn_attention_49/core_gcn_99/conv1d_397/conv1d/BatchToSpaceNDД
>gcn_attention_49/core_gcn_99/conv1d_397/BiasAdd/ReadVariableOpReadVariableOpGgcn_attention_49_core_gcn_99_conv1d_397_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_49/core_gcn_99/conv1d_397/BiasAdd/ReadVariableOpЉ
/gcn_attention_49/core_gcn_99/conv1d_397/BiasAddBiasAddFgcn_attention_49/core_gcn_99/conv1d_397/conv1d/BatchToSpaceND:output:0Fgcn_attention_49/core_gcn_99/conv1d_397/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€21
/gcn_attention_49/core_gcn_99/conv1d_397/BiasAddҐ
Hgcn_attention_49/core_gcn_99/batch_normalization_796/Cast/ReadVariableOpReadVariableOpQgcn_attention_49_core_gcn_99_batch_normalization_796_cast_readvariableop_resource*
_output_shapes
:*
dtype02J
Hgcn_attention_49/core_gcn_99/batch_normalization_796/Cast/ReadVariableOp®
Jgcn_attention_49/core_gcn_99/batch_normalization_796/Cast_1/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_99_batch_normalization_796_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_99/batch_normalization_796/Cast_1/ReadVariableOp®
Jgcn_attention_49/core_gcn_99/batch_normalization_796/Cast_2/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_99_batch_normalization_796_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_99/batch_normalization_796/Cast_2/ReadVariableOp®
Jgcn_attention_49/core_gcn_99/batch_normalization_796/Cast_3/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_99_batch_normalization_796_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_99/batch_normalization_796/Cast_3/ReadVariableOp—
Dgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2F
Dgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/add/yў
Bgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/addAddV2Rgcn_attention_49/core_gcn_99/batch_normalization_796/Cast_1/ReadVariableOp:value:0Mgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/add/y:output:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/addВ
Dgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/RsqrtRsqrtFgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/add:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/Rsqrt“
Bgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/mulMulHgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/Rsqrt:y:0Rgcn_attention_49/core_gcn_99/batch_normalization_796/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/mul‘
Dgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/mul_1Mul8gcn_attention_49/core_gcn_99/conv1d_397/BiasAdd:output:0Fgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2F
Dgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/mul_1“
Dgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/mul_2MulPgcn_attention_49/core_gcn_99/batch_normalization_796/Cast/ReadVariableOp:value:0Fgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/mul:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/mul_2“
Bgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/subSubRgcn_attention_49/core_gcn_99/batch_normalization_796/Cast_2/ReadVariableOp:value:0Hgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/subж
Dgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/add_1AddV2Hgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/mul_1:z:0Fgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2F
Dgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/add_1у
/gcn_attention_49/core_gcn_99/activation_10/ReluReluHgcn_attention_49/core_gcn_99/batch_normalization_796/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€21
/gcn_attention_49/core_gcn_99/activation_10/Reluр
1gcn_attention_49/core_gcn_99/dropout_598/IdentityIdentity=gcn_attention_49/core_gcn_99/activation_10/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€23
1gcn_attention_49/core_gcn_99/dropout_598/Identity∆
<gcn_attention_49/core_gcn_99/conv1d_398/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:2>
<gcn_attention_49/core_gcn_99/conv1d_398/conv1d/dilation_rateѕ
;gcn_attention_49/core_gcn_99/conv1d_398/conv1d/filter_shapeConst*
_output_shapes
:*
dtype0*!
valueB"         2=
;gcn_attention_49/core_gcn_99/conv1d_398/conv1d/filter_shape≈
4gcn_attention_49/core_gcn_99/conv1d_398/conv1d/stackConst*
_output_shapes

:*
dtype0*!
valueB"      26
4gcn_attention_49/core_gcn_99/conv1d_398/conv1d/stack∆
4gcn_attention_49/core_gcn_99/conv1d_398/conv1d/ShapeShape*gcn_attention_49/core_gcn_98/add_2/add:z:0*
T0*
_output_shapes
:26
4gcn_attention_49/core_gcn_99/conv1d_398/conv1d/Shape“
Bgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2D
Bgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice/stack÷
Dgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice/stack_1÷
Dgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice/stack_2ь
<gcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_sliceStridedSlice=gcn_attention_49/core_gcn_99/conv1d_398/conv1d/Shape:output:0Kgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice/stack:output:0Mgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice/stack_1:output:0Mgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<gcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_sliceн
6gcn_attention_49/core_gcn_99/conv1d_398/conv1d/stack_1PackEgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice:output:0*
N*
T0*
_output_shapes
:28
6gcn_attention_49/core_gcn_99/conv1d_398/conv1d/stack_1Ы
cgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2e
cgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice/stackЯ
egcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2g
egcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice/stack_1Я
egcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2g
egcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice/stack_2«
]gcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_sliceStridedSlice=gcn_attention_49/core_gcn_99/conv1d_398/conv1d/stack:output:0lgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice/stack:output:0ngcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice/stack_1:output:0ngcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2_
]gcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_sliceЯ
egcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2g
egcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_1/stack£
ggcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2i
ggcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1£
ggcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2i
ggcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2—
_gcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_1StridedSlice=gcn_attention_49/core_gcn_99/conv1d_398/conv1d/stack:output:0ngcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_1/stack:output:0pgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1:output:0pgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2a
_gcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_1Б
Sgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/addAddV2?gcn_attention_49/core_gcn_99/conv1d_398/conv1d/stack_1:output:0fgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice:output:0*
T0*
_output_shapes
:2U
Sgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/addЯ
Ugcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/add_1AddV2Wgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/add:z:0hgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_1:output:0*
T0*
_output_shapes
:2W
Ugcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/add_1э
Sgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/modFloorModYgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/add_1:z:0Egcn_attention_49/core_gcn_99/conv1d_398/conv1d/dilation_rate:output:0*
T0*
_output_shapes
:2U
Sgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/modц
Sgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/subSubEgcn_attention_49/core_gcn_99/conv1d_398/conv1d/dilation_rate:output:0Wgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/mod:z:0*
T0*
_output_shapes
:2U
Sgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/sub€
Ugcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/mod_1FloorModWgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/sub:z:0Egcn_attention_49/core_gcn_99/conv1d_398/conv1d/dilation_rate:output:0*
T0*
_output_shapes
:2W
Ugcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/mod_1°
Ugcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/add_2AddV2hgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_1:output:0Ygcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/mod_1:z:0*
T0*
_output_shapes
:2W
Ugcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/add_2Ш
egcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_2/stackЬ
ggcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1Ь
ggcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2‘
_gcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_2StridedSlicefgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice:output:0ngcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_2/stack:output:0pgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1:output:0pgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_2Ш
egcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_3/stackЬ
ggcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1Ь
ggcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2«
_gcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_3StridedSliceYgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/add_2:z:0ngcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_3/stack:output:0pgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1:output:0pgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_3¬
Zgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/paddings/0Packhgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_2:output:0hgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2\
Zgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/paddings/0”
Xgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/paddingsPackcgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/paddings/0:output:0*
N*
T0*
_output_shapes

:2Z
Xgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/paddingsШ
egcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2g
egcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_4/stackЬ
ggcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1Ь
ggcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
ggcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2«
_gcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_4StridedSliceYgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/mod_1:z:0ngcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_4/stack:output:0pgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1:output:0pgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2a
_gcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_4ш
Ygcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/crops/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2[
Ygcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/crops/0/0ґ
Wgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/crops/0Packbgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/crops/0/0:output:0hgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/strided_slice_4:output:0*
N*
T0*
_output_shapes
:2Y
Wgcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/crops/0 
Ugcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/cropsPack`gcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/crops/0:output:0*
N*
T0*
_output_shapes

:2W
Ugcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/crops÷
Dgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_1/stackЏ
Fgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_1/stack_1Џ
Fgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_1/stack_2Ъ
>gcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_1StridedSliceagcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/paddings:output:0Mgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_1/stack:output:0Ogcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_1/stack_1:output:0Ogcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:2@
>gcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_1∆
@gcn_attention_49/core_gcn_99/conv1d_398/conv1d/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 2B
@gcn_attention_49/core_gcn_99/conv1d_398/conv1d/concat/concat_dimъ
<gcn_attention_49/core_gcn_99/conv1d_398/conv1d/concat/concatIdentityGgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_1:output:0*
T0*
_output_shapes

:2>
<gcn_attention_49/core_gcn_99/conv1d_398/conv1d/concat/concat÷
Dgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_2/stackЏ
Fgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_2/stack_1Џ
Fgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_2/stack_2Ч
>gcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_2StridedSlice^gcn_attention_49/core_gcn_99/conv1d_398/conv1d/required_space_to_batch_paddings/crops:output:0Mgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_2/stack:output:0Ogcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_2/stack_1:output:0Ogcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:2@
>gcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_2 
Bgcn_attention_49/core_gcn_99/conv1d_398/conv1d/concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bgcn_attention_49/core_gcn_99/conv1d_398/conv1d/concat_1/concat_dimю
>gcn_attention_49/core_gcn_99/conv1d_398/conv1d/concat_1/concatIdentityGgcn_attention_49/core_gcn_99/conv1d_398/conv1d/strided_slice_2:output:0*
T0*
_output_shapes

:2@
>gcn_attention_49/core_gcn_99/conv1d_398/conv1d/concat_1/concatа
Igcn_attention_49/core_gcn_99/conv1d_398/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2K
Igcn_attention_49/core_gcn_99/conv1d_398/conv1d/SpaceToBatchND/block_shapeЦ
=gcn_attention_49/core_gcn_99/conv1d_398/conv1d/SpaceToBatchNDSpaceToBatchND*gcn_attention_49/core_gcn_98/add_2/add:z:0Rgcn_attention_49/core_gcn_99/conv1d_398/conv1d/SpaceToBatchND/block_shape:output:0Egcn_attention_49/core_gcn_99/conv1d_398/conv1d/concat/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2?
=gcn_attention_49/core_gcn_99/conv1d_398/conv1d/SpaceToBatchND…
=gcn_attention_49/core_gcn_99/conv1d_398/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2?
=gcn_attention_49/core_gcn_99/conv1d_398/conv1d/ExpandDims/dim„
9gcn_attention_49/core_gcn_99/conv1d_398/conv1d/ExpandDims
ExpandDimsFgcn_attention_49/core_gcn_99/conv1d_398/conv1d/SpaceToBatchND:output:0Fgcn_attention_49/core_gcn_99/conv1d_398/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2;
9gcn_attention_49/core_gcn_99/conv1d_398/conv1d/ExpandDims∞
Jgcn_attention_49/core_gcn_99/conv1d_398/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_99_conv1d_398_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_99/conv1d_398/conv1d/ExpandDims_1/ReadVariableOpƒ
?gcn_attention_49/core_gcn_99/conv1d_398/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2A
?gcn_attention_49/core_gcn_99/conv1d_398/conv1d/ExpandDims_1/dim„
;gcn_attention_49/core_gcn_99/conv1d_398/conv1d/ExpandDims_1
ExpandDimsRgcn_attention_49/core_gcn_99/conv1d_398/conv1d/ExpandDims_1/ReadVariableOp:value:0Hgcn_attention_49/core_gcn_99/conv1d_398/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2=
;gcn_attention_49/core_gcn_99/conv1d_398/conv1d/ExpandDims_1а
.gcn_attention_49/core_gcn_99/conv1d_398/conv1dConv2DBgcn_attention_49/core_gcn_99/conv1d_398/conv1d/ExpandDims:output:0Dgcn_attention_49/core_gcn_99/conv1d_398/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
paddingVALID*
strides
20
.gcn_attention_49/core_gcn_99/conv1d_398/conv1dУ
6gcn_attention_49/core_gcn_99/conv1d_398/conv1d/SqueezeSqueeze7gcn_attention_49/core_gcn_99/conv1d_398/conv1d:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
squeeze_dims

э€€€€€€€€28
6gcn_attention_49/core_gcn_99/conv1d_398/conv1d/Squeezeа
Igcn_attention_49/core_gcn_99/conv1d_398/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2K
Igcn_attention_49/core_gcn_99/conv1d_398/conv1d/BatchToSpaceND/block_shape≠
=gcn_attention_49/core_gcn_99/conv1d_398/conv1d/BatchToSpaceNDBatchToSpaceND?gcn_attention_49/core_gcn_99/conv1d_398/conv1d/Squeeze:output:0Rgcn_attention_49/core_gcn_99/conv1d_398/conv1d/BatchToSpaceND/block_shape:output:0Ggcn_attention_49/core_gcn_99/conv1d_398/conv1d/concat_1/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2?
=gcn_attention_49/core_gcn_99/conv1d_398/conv1d/BatchToSpaceNDД
>gcn_attention_49/core_gcn_99/conv1d_398/BiasAdd/ReadVariableOpReadVariableOpGgcn_attention_49_core_gcn_99_conv1d_398_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_49/core_gcn_99/conv1d_398/BiasAdd/ReadVariableOpЉ
/gcn_attention_49/core_gcn_99/conv1d_398/BiasAddBiasAddFgcn_attention_49/core_gcn_99/conv1d_398/conv1d/BatchToSpaceND:output:0Fgcn_attention_49/core_gcn_99/conv1d_398/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€21
/gcn_attention_49/core_gcn_99/conv1d_398/BiasAddҐ
Hgcn_attention_49/core_gcn_99/batch_normalization_797/Cast/ReadVariableOpReadVariableOpQgcn_attention_49_core_gcn_99_batch_normalization_797_cast_readvariableop_resource*
_output_shapes
:*
dtype02J
Hgcn_attention_49/core_gcn_99/batch_normalization_797/Cast/ReadVariableOp®
Jgcn_attention_49/core_gcn_99/batch_normalization_797/Cast_1/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_99_batch_normalization_797_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_99/batch_normalization_797/Cast_1/ReadVariableOp®
Jgcn_attention_49/core_gcn_99/batch_normalization_797/Cast_2/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_99_batch_normalization_797_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_99/batch_normalization_797/Cast_2/ReadVariableOp®
Jgcn_attention_49/core_gcn_99/batch_normalization_797/Cast_3/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_99_batch_normalization_797_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_99/batch_normalization_797/Cast_3/ReadVariableOp—
Dgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2F
Dgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/add/yў
Bgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/addAddV2Rgcn_attention_49/core_gcn_99/batch_normalization_797/Cast_1/ReadVariableOp:value:0Mgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/add/y:output:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/addВ
Dgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/RsqrtRsqrtFgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/add:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/Rsqrt“
Bgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/mulMulHgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/Rsqrt:y:0Rgcn_attention_49/core_gcn_99/batch_normalization_797/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/mul‘
Dgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/mul_1Mul8gcn_attention_49/core_gcn_99/conv1d_398/BiasAdd:output:0Fgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2F
Dgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/mul_1“
Dgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/mul_2MulPgcn_attention_49/core_gcn_99/batch_normalization_797/Cast/ReadVariableOp:value:0Fgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/mul:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/mul_2“
Bgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/subSubRgcn_attention_49/core_gcn_99/batch_normalization_797/Cast_2/ReadVariableOp:value:0Hgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/subж
Dgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/add_1AddV2Hgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/mul_1:z:0Fgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2F
Dgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/add_1у
/gcn_attention_49/core_gcn_99/activation_11/ReluReluHgcn_attention_49/core_gcn_99/batch_normalization_797/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€21
/gcn_attention_49/core_gcn_99/activation_11/Reluр
1gcn_attention_49/core_gcn_99/dropout_599/IdentityIdentity=gcn_attention_49/core_gcn_99/activation_11/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€23
1gcn_attention_49/core_gcn_99/dropout_599/Identity†
-gcn_attention_49/core_gcn_99/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-gcn_attention_49/core_gcn_99/ExpandDims_2/dimЫ
)gcn_attention_49/core_gcn_99/ExpandDims_2
ExpandDims:gcn_attention_49/core_gcn_99/dropout_599/Identity:output:06gcn_attention_49/core_gcn_99/ExpandDims_2/dim:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2+
)gcn_attention_49/core_gcn_99/ExpandDims_2Ш
+gcn_attention_49/core_gcn_99/multiply_1/mulMul5gcn_attention_49/core_gcn_99/activation_9/Sigmoid:y:02gcn_attention_49/core_gcn_99/ExpandDims_2:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2-
+gcn_attention_49/core_gcn_99/multiply_1/mulЉ
;gcn_attention_49/core_gcn_99/lambda_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;gcn_attention_49/core_gcn_99/lambda_2/Sum/reduction_indicesУ
)gcn_attention_49/core_gcn_99/lambda_2/SumSum/gcn_attention_49/core_gcn_99/multiply_1/mul:z:0Dgcn_attention_49/core_gcn_99/lambda_2/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2+
)gcn_attention_49/core_gcn_99/lambda_2/Sumј
=gcn_attention_49/core_gcn_99/lambda_2/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2?
=gcn_attention_49/core_gcn_99/lambda_2/Sum_1/reduction_indicesЯ
+gcn_attention_49/core_gcn_99/lambda_2/Sum_1Sum5gcn_attention_49/core_gcn_99/activation_9/Sigmoid:y:0Fgcn_attention_49/core_gcn_99/lambda_2/Sum_1/reduction_indices:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2-
+gcn_attention_49/core_gcn_99/lambda_2/Sum_1Я
+gcn_attention_49/core_gcn_99/lambda_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *е<2-
+gcn_attention_49/core_gcn_99/lambda_3/add/yК
)gcn_attention_49/core_gcn_99/lambda_3/addAddV24gcn_attention_49/core_gcn_99/lambda_2/Sum_1:output:04gcn_attention_49/core_gcn_99/lambda_3/add/y:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2+
)gcn_attention_49/core_gcn_99/lambda_3/addЛ
-gcn_attention_49/core_gcn_99/lambda_3/truedivRealDiv2gcn_attention_49/core_gcn_99/lambda_2/Sum:output:0-gcn_attention_49/core_gcn_99/lambda_3/add:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2/
-gcn_attention_49/core_gcn_99/lambda_3/truedivЗ
&gcn_attention_49/core_gcn_99/add_5/addAddV2:gcn_attention_49/core_gcn_99/dropout_598/Identity:output:01gcn_attention_49/core_gcn_99/lambda_3/truediv:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2(
&gcn_attention_49/core_gcn_99/add_5/addУ
Cgcn_attention_49/core_gcn_99/batch_normalization_798/ReadVariableOpReadVariableOpLgcn_attention_49_core_gcn_99_batch_normalization_798_readvariableop_resource*
_output_shapes
:*
dtype02E
Cgcn_attention_49/core_gcn_99/batch_normalization_798/ReadVariableOpЩ
Egcn_attention_49/core_gcn_99/batch_normalization_798/ReadVariableOp_1ReadVariableOpNgcn_attention_49_core_gcn_99_batch_normalization_798_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Egcn_attention_49/core_gcn_99/batch_normalization_798/ReadVariableOp_1∆
Tgcn_attention_49/core_gcn_99/batch_normalization_798/FusedBatchNormV3/ReadVariableOpReadVariableOp]gcn_attention_49_core_gcn_99_batch_normalization_798_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02V
Tgcn_attention_49/core_gcn_99/batch_normalization_798/FusedBatchNormV3/ReadVariableOpћ
Vgcn_attention_49/core_gcn_99/batch_normalization_798/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_gcn_attention_49_core_gcn_99_batch_normalization_798_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02X
Vgcn_attention_49/core_gcn_99/batch_normalization_798/FusedBatchNormV3/ReadVariableOp_1ј
Egcn_attention_49/core_gcn_99/batch_normalization_798/FusedBatchNormV3FusedBatchNormV3,gcn_attention_49/core_gcn_99/add_4/add_1:z:0Kgcn_attention_49/core_gcn_99/batch_normalization_798/ReadVariableOp:value:0Mgcn_attention_49/core_gcn_99/batch_normalization_798/ReadVariableOp_1:value:0\gcn_attention_49/core_gcn_99/batch_normalization_798/FusedBatchNormV3/ReadVariableOp:value:0^gcn_attention_49/core_gcn_99/batch_normalization_798/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2G
Egcn_attention_49/core_gcn_99/batch_normalization_798/FusedBatchNormV3Ґ
Hgcn_attention_49/core_gcn_99/batch_normalization_799/Cast/ReadVariableOpReadVariableOpQgcn_attention_49_core_gcn_99_batch_normalization_799_cast_readvariableop_resource*
_output_shapes
:*
dtype02J
Hgcn_attention_49/core_gcn_99/batch_normalization_799/Cast/ReadVariableOp®
Jgcn_attention_49/core_gcn_99/batch_normalization_799/Cast_1/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_99_batch_normalization_799_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_99/batch_normalization_799/Cast_1/ReadVariableOp®
Jgcn_attention_49/core_gcn_99/batch_normalization_799/Cast_2/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_99_batch_normalization_799_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_99/batch_normalization_799/Cast_2/ReadVariableOp®
Jgcn_attention_49/core_gcn_99/batch_normalization_799/Cast_3/ReadVariableOpReadVariableOpSgcn_attention_49_core_gcn_99_batch_normalization_799_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02L
Jgcn_attention_49/core_gcn_99/batch_normalization_799/Cast_3/ReadVariableOp—
Dgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2F
Dgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/add/yў
Bgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/addAddV2Rgcn_attention_49/core_gcn_99/batch_normalization_799/Cast_1/ReadVariableOp:value:0Mgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/add/y:output:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/addВ
Dgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/RsqrtRsqrtFgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/add:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/Rsqrt“
Bgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/mulMulHgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/Rsqrt:y:0Rgcn_attention_49/core_gcn_99/batch_normalization_799/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/mul∆
Dgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/mul_1Mul*gcn_attention_49/core_gcn_99/add_5/add:z:0Fgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2F
Dgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/mul_1“
Dgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/mul_2MulPgcn_attention_49/core_gcn_99/batch_normalization_799/Cast/ReadVariableOp:value:0Fgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/mul:z:0*
T0*
_output_shapes
:2F
Dgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/mul_2“
Bgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/subSubRgcn_attention_49/core_gcn_99/batch_normalization_799/Cast_2/ReadVariableOp:value:0Hgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2D
Bgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/subж
Dgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/add_1AddV2Hgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/mul_1:z:0Fgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2F
Dgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/add_1Б
/gcn_attention_49/core_gcn_99/activation_12/ReluReluIgcn_attention_49/core_gcn_99/batch_normalization_798/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€21
/gcn_attention_49/core_gcn_99/activation_12/Reluу
/gcn_attention_49/core_gcn_99/activation_13/ReluReluHgcn_attention_49/core_gcn_99/batch_normalization_799/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€21
/gcn_attention_49/core_gcn_99/activation_13/ReluГ
&gcn_attention_49/core_gcn_99/add_6/addAddV2*gcn_attention_49/core_gcn_98/add_2/add:z:0=gcn_attention_49/core_gcn_99/activation_13/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2(
&gcn_attention_49/core_gcn_99/add_6/addР
&gcn_attention_49/core_gcn_99/add_7/addAddV2*gcn_attention_49/core_gcn_98/add_3/add:z:0=gcn_attention_49/core_gcn_99/activation_12/Relu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2(
&gcn_attention_49/core_gcn_99/add_7/addњ
$gcn_attention_49/activation_248/ReluRelu*gcn_attention_49/core_gcn_99/add_6/add:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2&
$gcn_attention_49/activation_248/Reluю
<gcn_attention_49/batch_normalization_787/Cast/ReadVariableOpReadVariableOpEgcn_attention_49_batch_normalization_787_cast_readvariableop_resource*
_output_shapes
:*
dtype02>
<gcn_attention_49/batch_normalization_787/Cast/ReadVariableOpД
>gcn_attention_49/batch_normalization_787/Cast_1/ReadVariableOpReadVariableOpGgcn_attention_49_batch_normalization_787_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_49/batch_normalization_787/Cast_1/ReadVariableOpД
>gcn_attention_49/batch_normalization_787/Cast_2/ReadVariableOpReadVariableOpGgcn_attention_49_batch_normalization_787_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_49/batch_normalization_787/Cast_2/ReadVariableOpД
>gcn_attention_49/batch_normalization_787/Cast_3/ReadVariableOpReadVariableOpGgcn_attention_49_batch_normalization_787_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02@
>gcn_attention_49/batch_normalization_787/Cast_3/ReadVariableOpє
8gcn_attention_49/batch_normalization_787/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2:
8gcn_attention_49/batch_normalization_787/batchnorm/add/y©
6gcn_attention_49/batch_normalization_787/batchnorm/addAddV2Fgcn_attention_49/batch_normalization_787/Cast_1/ReadVariableOp:value:0Agcn_attention_49/batch_normalization_787/batchnorm/add/y:output:0*
T0*
_output_shapes
:28
6gcn_attention_49/batch_normalization_787/batchnorm/addё
8gcn_attention_49/batch_normalization_787/batchnorm/RsqrtRsqrt:gcn_attention_49/batch_normalization_787/batchnorm/add:z:0*
T0*
_output_shapes
:2:
8gcn_attention_49/batch_normalization_787/batchnorm/RsqrtҐ
6gcn_attention_49/batch_normalization_787/batchnorm/mulMul<gcn_attention_49/batch_normalization_787/batchnorm/Rsqrt:y:0Fgcn_attention_49/batch_normalization_787/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:28
6gcn_attention_49/batch_normalization_787/batchnorm/mul™
8gcn_attention_49/batch_normalization_787/batchnorm/mul_1Mul2gcn_attention_49/activation_248/Relu:activations:0:gcn_attention_49/batch_normalization_787/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2:
8gcn_attention_49/batch_normalization_787/batchnorm/mul_1Ґ
8gcn_attention_49/batch_normalization_787/batchnorm/mul_2MulDgcn_attention_49/batch_normalization_787/Cast/ReadVariableOp:value:0:gcn_attention_49/batch_normalization_787/batchnorm/mul:z:0*
T0*
_output_shapes
:2:
8gcn_attention_49/batch_normalization_787/batchnorm/mul_2Ґ
6gcn_attention_49/batch_normalization_787/batchnorm/subSubFgcn_attention_49/batch_normalization_787/Cast_2/ReadVariableOp:value:0<gcn_attention_49/batch_normalization_787/batchnorm/mul_2:z:0*
T0*
_output_shapes
:28
6gcn_attention_49/batch_normalization_787/batchnorm/subґ
8gcn_attention_49/batch_normalization_787/batchnorm/add_1AddV2<gcn_attention_49/batch_normalization_787/batchnorm/mul_1:z:0:gcn_attention_49/batch_normalization_787/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2:
8gcn_attention_49/batch_normalization_787/batchnorm/add_1„
%gcn_attention_49/dropout_591/IdentityIdentity<gcn_attention_49/batch_normalization_787/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2'
%gcn_attention_49/dropout_591/IdentityЃ
0gcn_attention_49/conv1d_399/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:22
0gcn_attention_49/conv1d_399/conv1d/dilation_rateЈ
/gcn_attention_49/conv1d_399/conv1d/filter_shapeConst*
_output_shapes
:*
dtype0*!
valueB"         21
/gcn_attention_49/conv1d_399/conv1d/filter_shape≠
(gcn_attention_49/conv1d_399/conv1d/stackConst*
_output_shapes

:*
dtype0*!
valueB"      2*
(gcn_attention_49/conv1d_399/conv1d/stack≤
(gcn_attention_49/conv1d_399/conv1d/ShapeShape.gcn_attention_49/dropout_591/Identity:output:0*
T0*
_output_shapes
:2*
(gcn_attention_49/conv1d_399/conv1d/ShapeЇ
6gcn_attention_49/conv1d_399/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6gcn_attention_49/conv1d_399/conv1d/strided_slice/stackЊ
8gcn_attention_49/conv1d_399/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8gcn_attention_49/conv1d_399/conv1d/strided_slice/stack_1Њ
8gcn_attention_49/conv1d_399/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8gcn_attention_49/conv1d_399/conv1d/strided_slice/stack_2і
0gcn_attention_49/conv1d_399/conv1d/strided_sliceStridedSlice1gcn_attention_49/conv1d_399/conv1d/Shape:output:0?gcn_attention_49/conv1d_399/conv1d/strided_slice/stack:output:0Agcn_attention_49/conv1d_399/conv1d/strided_slice/stack_1:output:0Agcn_attention_49/conv1d_399/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0gcn_attention_49/conv1d_399/conv1d/strided_slice…
*gcn_attention_49/conv1d_399/conv1d/stack_1Pack9gcn_attention_49/conv1d_399/conv1d/strided_slice:output:0*
N*
T0*
_output_shapes
:2,
*gcn_attention_49/conv1d_399/conv1d/stack_1Г
Wgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2Y
Wgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice/stackЗ
Ygcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2[
Ygcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice/stack_1З
Ygcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2[
Ygcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice/stack_2€
Qgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_sliceStridedSlice1gcn_attention_49/conv1d_399/conv1d/stack:output:0`gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice/stack:output:0bgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice/stack_1:output:0bgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2S
Qgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_sliceЗ
Ygcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2[
Ygcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_1/stackЛ
[gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2]
[gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1Л
[gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2]
[gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2Й
Sgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_1StridedSlice1gcn_attention_49/conv1d_399/conv1d/stack:output:0bgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_1/stack:output:0dgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_1:output:0dgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2U
Sgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_1—
Ggcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/addAddV23gcn_attention_49/conv1d_399/conv1d/stack_1:output:0Zgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice:output:0*
T0*
_output_shapes
:2I
Ggcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/addп
Igcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/add_1AddV2Kgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/add:z:0\gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_1:output:0*
T0*
_output_shapes
:2K
Igcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/add_1Ќ
Ggcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/modFloorModMgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/add_1:z:09gcn_attention_49/conv1d_399/conv1d/dilation_rate:output:0*
T0*
_output_shapes
:2I
Ggcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/mod∆
Ggcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/subSub9gcn_attention_49/conv1d_399/conv1d/dilation_rate:output:0Kgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/mod:z:0*
T0*
_output_shapes
:2I
Ggcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/subѕ
Igcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/mod_1FloorModKgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/sub:z:09gcn_attention_49/conv1d_399/conv1d/dilation_rate:output:0*
T0*
_output_shapes
:2K
Igcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/mod_1с
Igcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/add_2AddV2\gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_1:output:0Mgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/mod_1:z:0*
T0*
_output_shapes
:2K
Igcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/add_2А
Ygcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2[
Ygcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_2/stackД
[gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2]
[gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1Д
[gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2]
[gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2М
Sgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_2StridedSliceZgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice:output:0bgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_2/stack:output:0dgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_1:output:0dgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2U
Sgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_2А
Ygcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2[
Ygcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_3/stackД
[gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2]
[gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1Д
[gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2]
[gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2€
Sgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_3StridedSliceMgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/add_2:z:0bgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_3/stack:output:0dgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_1:output:0dgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2U
Sgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_3Т
Ngcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/paddings/0Pack\gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_2:output:0\gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2P
Ngcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/paddings/0ѓ
Lgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/paddingsPackWgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/paddings/0:output:0*
N*
T0*
_output_shapes

:2N
Lgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/paddingsА
Ygcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2[
Ygcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_4/stackД
[gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2]
[gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1Д
[gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2]
[gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2€
Sgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_4StridedSliceMgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/mod_1:z:0bgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_4/stack:output:0dgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_1:output:0dgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2U
Sgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_4а
Mgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/crops/0/0Const*
_output_shapes
: *
dtype0*
value	B : 2O
Mgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/crops/0/0Ж
Kgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/crops/0PackVgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/crops/0/0:output:0\gcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/strided_slice_4:output:0*
N*
T0*
_output_shapes
:2M
Kgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/crops/0¶
Igcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/cropsPackTgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/crops/0:output:0*
N*
T0*
_output_shapes

:2K
Igcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/cropsЊ
8gcn_attention_49/conv1d_399/conv1d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8gcn_attention_49/conv1d_399/conv1d/strided_slice_1/stack¬
:gcn_attention_49/conv1d_399/conv1d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:gcn_attention_49/conv1d_399/conv1d/strided_slice_1/stack_1¬
:gcn_attention_49/conv1d_399/conv1d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:gcn_attention_49/conv1d_399/conv1d/strided_slice_1/stack_2“
2gcn_attention_49/conv1d_399/conv1d/strided_slice_1StridedSliceUgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/paddings:output:0Agcn_attention_49/conv1d_399/conv1d/strided_slice_1/stack:output:0Cgcn_attention_49/conv1d_399/conv1d/strided_slice_1/stack_1:output:0Cgcn_attention_49/conv1d_399/conv1d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:24
2gcn_attention_49/conv1d_399/conv1d/strided_slice_1Ѓ
4gcn_attention_49/conv1d_399/conv1d/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4gcn_attention_49/conv1d_399/conv1d/concat/concat_dim÷
0gcn_attention_49/conv1d_399/conv1d/concat/concatIdentity;gcn_attention_49/conv1d_399/conv1d/strided_slice_1:output:0*
T0*
_output_shapes

:22
0gcn_attention_49/conv1d_399/conv1d/concat/concatЊ
8gcn_attention_49/conv1d_399/conv1d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8gcn_attention_49/conv1d_399/conv1d/strided_slice_2/stack¬
:gcn_attention_49/conv1d_399/conv1d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:gcn_attention_49/conv1d_399/conv1d/strided_slice_2/stack_1¬
:gcn_attention_49/conv1d_399/conv1d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:gcn_attention_49/conv1d_399/conv1d/strided_slice_2/stack_2ѕ
2gcn_attention_49/conv1d_399/conv1d/strided_slice_2StridedSliceRgcn_attention_49/conv1d_399/conv1d/required_space_to_batch_paddings/crops:output:0Agcn_attention_49/conv1d_399/conv1d/strided_slice_2/stack:output:0Cgcn_attention_49/conv1d_399/conv1d/strided_slice_2/stack_1:output:0Cgcn_attention_49/conv1d_399/conv1d/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:24
2gcn_attention_49/conv1d_399/conv1d/strided_slice_2≤
6gcn_attention_49/conv1d_399/conv1d/concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 28
6gcn_attention_49/conv1d_399/conv1d/concat_1/concat_dimЏ
2gcn_attention_49/conv1d_399/conv1d/concat_1/concatIdentity;gcn_attention_49/conv1d_399/conv1d/strided_slice_2:output:0*
T0*
_output_shapes

:24
2gcn_attention_49/conv1d_399/conv1d/concat_1/concat»
=gcn_attention_49/conv1d_399/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2?
=gcn_attention_49/conv1d_399/conv1d/SpaceToBatchND/block_shapeк
1gcn_attention_49/conv1d_399/conv1d/SpaceToBatchNDSpaceToBatchND.gcn_attention_49/dropout_591/Identity:output:0Fgcn_attention_49/conv1d_399/conv1d/SpaceToBatchND/block_shape:output:09gcn_attention_49/conv1d_399/conv1d/concat/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€23
1gcn_attention_49/conv1d_399/conv1d/SpaceToBatchND±
1gcn_attention_49/conv1d_399/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€23
1gcn_attention_49/conv1d_399/conv1d/ExpandDims/dimІ
-gcn_attention_49/conv1d_399/conv1d/ExpandDims
ExpandDims:gcn_attention_49/conv1d_399/conv1d/SpaceToBatchND:output:0:gcn_attention_49/conv1d_399/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2/
-gcn_attention_49/conv1d_399/conv1d/ExpandDimsМ
>gcn_attention_49/conv1d_399/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpGgcn_attention_49_conv1d_399_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02@
>gcn_attention_49/conv1d_399/conv1d/ExpandDims_1/ReadVariableOpђ
3gcn_attention_49/conv1d_399/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3gcn_attention_49/conv1d_399/conv1d/ExpandDims_1/dimІ
/gcn_attention_49/conv1d_399/conv1d/ExpandDims_1
ExpandDimsFgcn_attention_49/conv1d_399/conv1d/ExpandDims_1/ReadVariableOp:value:0<gcn_attention_49/conv1d_399/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:21
/gcn_attention_49/conv1d_399/conv1d/ExpandDims_1∞
"gcn_attention_49/conv1d_399/conv1dConv2D6gcn_attention_49/conv1d_399/conv1d/ExpandDims:output:08gcn_attention_49/conv1d_399/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
paddingVALID*
strides
2$
"gcn_attention_49/conv1d_399/conv1dп
*gcn_attention_49/conv1d_399/conv1d/SqueezeSqueeze+gcn_attention_49/conv1d_399/conv1d:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
squeeze_dims

э€€€€€€€€2,
*gcn_attention_49/conv1d_399/conv1d/Squeeze»
=gcn_attention_49/conv1d_399/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2?
=gcn_attention_49/conv1d_399/conv1d/BatchToSpaceND/block_shapeс
1gcn_attention_49/conv1d_399/conv1d/BatchToSpaceNDBatchToSpaceND3gcn_attention_49/conv1d_399/conv1d/Squeeze:output:0Fgcn_attention_49/conv1d_399/conv1d/BatchToSpaceND/block_shape:output:0;gcn_attention_49/conv1d_399/conv1d/concat_1/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€23
1gcn_attention_49/conv1d_399/conv1d/BatchToSpaceNDа
2gcn_attention_49/conv1d_399/BiasAdd/ReadVariableOpReadVariableOp;gcn_attention_49_conv1d_399_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2gcn_attention_49/conv1d_399/BiasAdd/ReadVariableOpМ
#gcn_attention_49/conv1d_399/BiasAddBiasAdd:gcn_attention_49/conv1d_399/conv1d/BatchToSpaceND:output:0:gcn_attention_49/conv1d_399/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2%
#gcn_attention_49/conv1d_399/BiasAddK
xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
xЩ
NotEqualNotEqualinput_text1
x:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ*
incompatible_shape_error( 2

NotEqualy
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
Any/reduction_indicesq
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
AnyФ
IdentityIdentity,gcn_attention_49/conv1d_399/BiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityщ/
NoOpNoOp=^gcn_attention_49/batch_normalization_784/Cast/ReadVariableOp?^gcn_attention_49/batch_normalization_784/Cast_1/ReadVariableOp?^gcn_attention_49/batch_normalization_784/Cast_2/ReadVariableOp?^gcn_attention_49/batch_normalization_784/Cast_3/ReadVariableOpI^gcn_attention_49/batch_normalization_786/FusedBatchNormV3/ReadVariableOpK^gcn_attention_49/batch_normalization_786/FusedBatchNormV3/ReadVariableOp_18^gcn_attention_49/batch_normalization_786/ReadVariableOp:^gcn_attention_49/batch_normalization_786/ReadVariableOp_1=^gcn_attention_49/batch_normalization_787/Cast/ReadVariableOp?^gcn_attention_49/batch_normalization_787/Cast_1/ReadVariableOp?^gcn_attention_49/batch_normalization_787/Cast_2/ReadVariableOp?^gcn_attention_49/batch_normalization_787/Cast_3/ReadVariableOp3^gcn_attention_49/conv1d_392/BiasAdd/ReadVariableOp?^gcn_attention_49/conv1d_392/conv1d/ExpandDims_1/ReadVariableOp3^gcn_attention_49/conv1d_399/BiasAdd/ReadVariableOp?^gcn_attention_49/conv1d_399/conv1d/ExpandDims_1/ReadVariableOp3^gcn_attention_49/conv2d_197/BiasAdd/ReadVariableOp2^gcn_attention_49/conv2d_197/Conv2D/ReadVariableOpU^gcn_attention_49/core_gcn_98/batch_normalization_788/FusedBatchNormV3/ReadVariableOpW^gcn_attention_49/core_gcn_98/batch_normalization_788/FusedBatchNormV3/ReadVariableOp_1D^gcn_attention_49/core_gcn_98/batch_normalization_788/ReadVariableOpF^gcn_attention_49/core_gcn_98/batch_normalization_788/ReadVariableOp_1I^gcn_attention_49/core_gcn_98/batch_normalization_789/Cast/ReadVariableOpK^gcn_attention_49/core_gcn_98/batch_normalization_789/Cast_1/ReadVariableOpK^gcn_attention_49/core_gcn_98/batch_normalization_789/Cast_2/ReadVariableOpK^gcn_attention_49/core_gcn_98/batch_normalization_789/Cast_3/ReadVariableOpI^gcn_attention_49/core_gcn_98/batch_normalization_790/Cast/ReadVariableOpK^gcn_attention_49/core_gcn_98/batch_normalization_790/Cast_1/ReadVariableOpK^gcn_attention_49/core_gcn_98/batch_normalization_790/Cast_2/ReadVariableOpK^gcn_attention_49/core_gcn_98/batch_normalization_790/Cast_3/ReadVariableOpI^gcn_attention_49/core_gcn_98/batch_normalization_791/Cast/ReadVariableOpK^gcn_attention_49/core_gcn_98/batch_normalization_791/Cast_1/ReadVariableOpK^gcn_attention_49/core_gcn_98/batch_normalization_791/Cast_2/ReadVariableOpK^gcn_attention_49/core_gcn_98/batch_normalization_791/Cast_3/ReadVariableOpU^gcn_attention_49/core_gcn_98/batch_normalization_792/FusedBatchNormV3/ReadVariableOpW^gcn_attention_49/core_gcn_98/batch_normalization_792/FusedBatchNormV3/ReadVariableOp_1D^gcn_attention_49/core_gcn_98/batch_normalization_792/ReadVariableOpF^gcn_attention_49/core_gcn_98/batch_normalization_792/ReadVariableOp_1I^gcn_attention_49/core_gcn_98/batch_normalization_793/Cast/ReadVariableOpK^gcn_attention_49/core_gcn_98/batch_normalization_793/Cast_1/ReadVariableOpK^gcn_attention_49/core_gcn_98/batch_normalization_793/Cast_2/ReadVariableOpK^gcn_attention_49/core_gcn_98/batch_normalization_793/Cast_3/ReadVariableOp?^gcn_attention_49/core_gcn_98/conv1d_393/BiasAdd/ReadVariableOpK^gcn_attention_49/core_gcn_98/conv1d_393/conv1d/ExpandDims_1/ReadVariableOp?^gcn_attention_49/core_gcn_98/conv1d_394/BiasAdd/ReadVariableOpK^gcn_attention_49/core_gcn_98/conv1d_394/conv1d/ExpandDims_1/ReadVariableOp?^gcn_attention_49/core_gcn_98/conv1d_395/BiasAdd/ReadVariableOpK^gcn_attention_49/core_gcn_98/conv1d_395/conv1d/ExpandDims_1/ReadVariableOp?^gcn_attention_49/core_gcn_98/conv2d_198/BiasAdd/ReadVariableOp>^gcn_attention_49/core_gcn_98/conv2d_198/Conv2D/ReadVariableOpU^gcn_attention_49/core_gcn_99/batch_normalization_794/FusedBatchNormV3/ReadVariableOpW^gcn_attention_49/core_gcn_99/batch_normalization_794/FusedBatchNormV3/ReadVariableOp_1D^gcn_attention_49/core_gcn_99/batch_normalization_794/ReadVariableOpF^gcn_attention_49/core_gcn_99/batch_normalization_794/ReadVariableOp_1I^gcn_attention_49/core_gcn_99/batch_normalization_795/Cast/ReadVariableOpK^gcn_attention_49/core_gcn_99/batch_normalization_795/Cast_1/ReadVariableOpK^gcn_attention_49/core_gcn_99/batch_normalization_795/Cast_2/ReadVariableOpK^gcn_attention_49/core_gcn_99/batch_normalization_795/Cast_3/ReadVariableOpI^gcn_attention_49/core_gcn_99/batch_normalization_796/Cast/ReadVariableOpK^gcn_attention_49/core_gcn_99/batch_normalization_796/Cast_1/ReadVariableOpK^gcn_attention_49/core_gcn_99/batch_normalization_796/Cast_2/ReadVariableOpK^gcn_attention_49/core_gcn_99/batch_normalization_796/Cast_3/ReadVariableOpI^gcn_attention_49/core_gcn_99/batch_normalization_797/Cast/ReadVariableOpK^gcn_attention_49/core_gcn_99/batch_normalization_797/Cast_1/ReadVariableOpK^gcn_attention_49/core_gcn_99/batch_normalization_797/Cast_2/ReadVariableOpK^gcn_attention_49/core_gcn_99/batch_normalization_797/Cast_3/ReadVariableOpU^gcn_attention_49/core_gcn_99/batch_normalization_798/FusedBatchNormV3/ReadVariableOpW^gcn_attention_49/core_gcn_99/batch_normalization_798/FusedBatchNormV3/ReadVariableOp_1D^gcn_attention_49/core_gcn_99/batch_normalization_798/ReadVariableOpF^gcn_attention_49/core_gcn_99/batch_normalization_798/ReadVariableOp_1I^gcn_attention_49/core_gcn_99/batch_normalization_799/Cast/ReadVariableOpK^gcn_attention_49/core_gcn_99/batch_normalization_799/Cast_1/ReadVariableOpK^gcn_attention_49/core_gcn_99/batch_normalization_799/Cast_2/ReadVariableOpK^gcn_attention_49/core_gcn_99/batch_normalization_799/Cast_3/ReadVariableOp?^gcn_attention_49/core_gcn_99/conv1d_396/BiasAdd/ReadVariableOpK^gcn_attention_49/core_gcn_99/conv1d_396/conv1d/ExpandDims_1/ReadVariableOp?^gcn_attention_49/core_gcn_99/conv1d_397/BiasAdd/ReadVariableOpK^gcn_attention_49/core_gcn_99/conv1d_397/conv1d/ExpandDims_1/ReadVariableOp?^gcn_attention_49/core_gcn_99/conv1d_398/BiasAdd/ReadVariableOpK^gcn_attention_49/core_gcn_99/conv1d_398/conv1d/ExpandDims_1/ReadVariableOp?^gcn_attention_49/core_gcn_99/conv2d_199/BiasAdd/ReadVariableOp>^gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/ReadVariableOp3^gcn_attention_49/dense_49/Tensordot/ReadVariableOp/^gcn_attention_49/embedding_49/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Є
_input_shapes¶
£:€€€€€€€€€€€€€€€€€€ђ:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2|
<gcn_attention_49/batch_normalization_784/Cast/ReadVariableOp<gcn_attention_49/batch_normalization_784/Cast/ReadVariableOp2А
>gcn_attention_49/batch_normalization_784/Cast_1/ReadVariableOp>gcn_attention_49/batch_normalization_784/Cast_1/ReadVariableOp2А
>gcn_attention_49/batch_normalization_784/Cast_2/ReadVariableOp>gcn_attention_49/batch_normalization_784/Cast_2/ReadVariableOp2А
>gcn_attention_49/batch_normalization_784/Cast_3/ReadVariableOp>gcn_attention_49/batch_normalization_784/Cast_3/ReadVariableOp2Ф
Hgcn_attention_49/batch_normalization_786/FusedBatchNormV3/ReadVariableOpHgcn_attention_49/batch_normalization_786/FusedBatchNormV3/ReadVariableOp2Ш
Jgcn_attention_49/batch_normalization_786/FusedBatchNormV3/ReadVariableOp_1Jgcn_attention_49/batch_normalization_786/FusedBatchNormV3/ReadVariableOp_12r
7gcn_attention_49/batch_normalization_786/ReadVariableOp7gcn_attention_49/batch_normalization_786/ReadVariableOp2v
9gcn_attention_49/batch_normalization_786/ReadVariableOp_19gcn_attention_49/batch_normalization_786/ReadVariableOp_12|
<gcn_attention_49/batch_normalization_787/Cast/ReadVariableOp<gcn_attention_49/batch_normalization_787/Cast/ReadVariableOp2А
>gcn_attention_49/batch_normalization_787/Cast_1/ReadVariableOp>gcn_attention_49/batch_normalization_787/Cast_1/ReadVariableOp2А
>gcn_attention_49/batch_normalization_787/Cast_2/ReadVariableOp>gcn_attention_49/batch_normalization_787/Cast_2/ReadVariableOp2А
>gcn_attention_49/batch_normalization_787/Cast_3/ReadVariableOp>gcn_attention_49/batch_normalization_787/Cast_3/ReadVariableOp2h
2gcn_attention_49/conv1d_392/BiasAdd/ReadVariableOp2gcn_attention_49/conv1d_392/BiasAdd/ReadVariableOp2А
>gcn_attention_49/conv1d_392/conv1d/ExpandDims_1/ReadVariableOp>gcn_attention_49/conv1d_392/conv1d/ExpandDims_1/ReadVariableOp2h
2gcn_attention_49/conv1d_399/BiasAdd/ReadVariableOp2gcn_attention_49/conv1d_399/BiasAdd/ReadVariableOp2А
>gcn_attention_49/conv1d_399/conv1d/ExpandDims_1/ReadVariableOp>gcn_attention_49/conv1d_399/conv1d/ExpandDims_1/ReadVariableOp2h
2gcn_attention_49/conv2d_197/BiasAdd/ReadVariableOp2gcn_attention_49/conv2d_197/BiasAdd/ReadVariableOp2f
1gcn_attention_49/conv2d_197/Conv2D/ReadVariableOp1gcn_attention_49/conv2d_197/Conv2D/ReadVariableOp2ђ
Tgcn_attention_49/core_gcn_98/batch_normalization_788/FusedBatchNormV3/ReadVariableOpTgcn_attention_49/core_gcn_98/batch_normalization_788/FusedBatchNormV3/ReadVariableOp2∞
Vgcn_attention_49/core_gcn_98/batch_normalization_788/FusedBatchNormV3/ReadVariableOp_1Vgcn_attention_49/core_gcn_98/batch_normalization_788/FusedBatchNormV3/ReadVariableOp_12К
Cgcn_attention_49/core_gcn_98/batch_normalization_788/ReadVariableOpCgcn_attention_49/core_gcn_98/batch_normalization_788/ReadVariableOp2О
Egcn_attention_49/core_gcn_98/batch_normalization_788/ReadVariableOp_1Egcn_attention_49/core_gcn_98/batch_normalization_788/ReadVariableOp_12Ф
Hgcn_attention_49/core_gcn_98/batch_normalization_789/Cast/ReadVariableOpHgcn_attention_49/core_gcn_98/batch_normalization_789/Cast/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_98/batch_normalization_789/Cast_1/ReadVariableOpJgcn_attention_49/core_gcn_98/batch_normalization_789/Cast_1/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_98/batch_normalization_789/Cast_2/ReadVariableOpJgcn_attention_49/core_gcn_98/batch_normalization_789/Cast_2/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_98/batch_normalization_789/Cast_3/ReadVariableOpJgcn_attention_49/core_gcn_98/batch_normalization_789/Cast_3/ReadVariableOp2Ф
Hgcn_attention_49/core_gcn_98/batch_normalization_790/Cast/ReadVariableOpHgcn_attention_49/core_gcn_98/batch_normalization_790/Cast/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_98/batch_normalization_790/Cast_1/ReadVariableOpJgcn_attention_49/core_gcn_98/batch_normalization_790/Cast_1/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_98/batch_normalization_790/Cast_2/ReadVariableOpJgcn_attention_49/core_gcn_98/batch_normalization_790/Cast_2/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_98/batch_normalization_790/Cast_3/ReadVariableOpJgcn_attention_49/core_gcn_98/batch_normalization_790/Cast_3/ReadVariableOp2Ф
Hgcn_attention_49/core_gcn_98/batch_normalization_791/Cast/ReadVariableOpHgcn_attention_49/core_gcn_98/batch_normalization_791/Cast/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_98/batch_normalization_791/Cast_1/ReadVariableOpJgcn_attention_49/core_gcn_98/batch_normalization_791/Cast_1/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_98/batch_normalization_791/Cast_2/ReadVariableOpJgcn_attention_49/core_gcn_98/batch_normalization_791/Cast_2/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_98/batch_normalization_791/Cast_3/ReadVariableOpJgcn_attention_49/core_gcn_98/batch_normalization_791/Cast_3/ReadVariableOp2ђ
Tgcn_attention_49/core_gcn_98/batch_normalization_792/FusedBatchNormV3/ReadVariableOpTgcn_attention_49/core_gcn_98/batch_normalization_792/FusedBatchNormV3/ReadVariableOp2∞
Vgcn_attention_49/core_gcn_98/batch_normalization_792/FusedBatchNormV3/ReadVariableOp_1Vgcn_attention_49/core_gcn_98/batch_normalization_792/FusedBatchNormV3/ReadVariableOp_12К
Cgcn_attention_49/core_gcn_98/batch_normalization_792/ReadVariableOpCgcn_attention_49/core_gcn_98/batch_normalization_792/ReadVariableOp2О
Egcn_attention_49/core_gcn_98/batch_normalization_792/ReadVariableOp_1Egcn_attention_49/core_gcn_98/batch_normalization_792/ReadVariableOp_12Ф
Hgcn_attention_49/core_gcn_98/batch_normalization_793/Cast/ReadVariableOpHgcn_attention_49/core_gcn_98/batch_normalization_793/Cast/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_98/batch_normalization_793/Cast_1/ReadVariableOpJgcn_attention_49/core_gcn_98/batch_normalization_793/Cast_1/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_98/batch_normalization_793/Cast_2/ReadVariableOpJgcn_attention_49/core_gcn_98/batch_normalization_793/Cast_2/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_98/batch_normalization_793/Cast_3/ReadVariableOpJgcn_attention_49/core_gcn_98/batch_normalization_793/Cast_3/ReadVariableOp2А
>gcn_attention_49/core_gcn_98/conv1d_393/BiasAdd/ReadVariableOp>gcn_attention_49/core_gcn_98/conv1d_393/BiasAdd/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_98/conv1d_393/conv1d/ExpandDims_1/ReadVariableOpJgcn_attention_49/core_gcn_98/conv1d_393/conv1d/ExpandDims_1/ReadVariableOp2А
>gcn_attention_49/core_gcn_98/conv1d_394/BiasAdd/ReadVariableOp>gcn_attention_49/core_gcn_98/conv1d_394/BiasAdd/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_98/conv1d_394/conv1d/ExpandDims_1/ReadVariableOpJgcn_attention_49/core_gcn_98/conv1d_394/conv1d/ExpandDims_1/ReadVariableOp2А
>gcn_attention_49/core_gcn_98/conv1d_395/BiasAdd/ReadVariableOp>gcn_attention_49/core_gcn_98/conv1d_395/BiasAdd/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_98/conv1d_395/conv1d/ExpandDims_1/ReadVariableOpJgcn_attention_49/core_gcn_98/conv1d_395/conv1d/ExpandDims_1/ReadVariableOp2А
>gcn_attention_49/core_gcn_98/conv2d_198/BiasAdd/ReadVariableOp>gcn_attention_49/core_gcn_98/conv2d_198/BiasAdd/ReadVariableOp2~
=gcn_attention_49/core_gcn_98/conv2d_198/Conv2D/ReadVariableOp=gcn_attention_49/core_gcn_98/conv2d_198/Conv2D/ReadVariableOp2ђ
Tgcn_attention_49/core_gcn_99/batch_normalization_794/FusedBatchNormV3/ReadVariableOpTgcn_attention_49/core_gcn_99/batch_normalization_794/FusedBatchNormV3/ReadVariableOp2∞
Vgcn_attention_49/core_gcn_99/batch_normalization_794/FusedBatchNormV3/ReadVariableOp_1Vgcn_attention_49/core_gcn_99/batch_normalization_794/FusedBatchNormV3/ReadVariableOp_12К
Cgcn_attention_49/core_gcn_99/batch_normalization_794/ReadVariableOpCgcn_attention_49/core_gcn_99/batch_normalization_794/ReadVariableOp2О
Egcn_attention_49/core_gcn_99/batch_normalization_794/ReadVariableOp_1Egcn_attention_49/core_gcn_99/batch_normalization_794/ReadVariableOp_12Ф
Hgcn_attention_49/core_gcn_99/batch_normalization_795/Cast/ReadVariableOpHgcn_attention_49/core_gcn_99/batch_normalization_795/Cast/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_99/batch_normalization_795/Cast_1/ReadVariableOpJgcn_attention_49/core_gcn_99/batch_normalization_795/Cast_1/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_99/batch_normalization_795/Cast_2/ReadVariableOpJgcn_attention_49/core_gcn_99/batch_normalization_795/Cast_2/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_99/batch_normalization_795/Cast_3/ReadVariableOpJgcn_attention_49/core_gcn_99/batch_normalization_795/Cast_3/ReadVariableOp2Ф
Hgcn_attention_49/core_gcn_99/batch_normalization_796/Cast/ReadVariableOpHgcn_attention_49/core_gcn_99/batch_normalization_796/Cast/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_99/batch_normalization_796/Cast_1/ReadVariableOpJgcn_attention_49/core_gcn_99/batch_normalization_796/Cast_1/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_99/batch_normalization_796/Cast_2/ReadVariableOpJgcn_attention_49/core_gcn_99/batch_normalization_796/Cast_2/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_99/batch_normalization_796/Cast_3/ReadVariableOpJgcn_attention_49/core_gcn_99/batch_normalization_796/Cast_3/ReadVariableOp2Ф
Hgcn_attention_49/core_gcn_99/batch_normalization_797/Cast/ReadVariableOpHgcn_attention_49/core_gcn_99/batch_normalization_797/Cast/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_99/batch_normalization_797/Cast_1/ReadVariableOpJgcn_attention_49/core_gcn_99/batch_normalization_797/Cast_1/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_99/batch_normalization_797/Cast_2/ReadVariableOpJgcn_attention_49/core_gcn_99/batch_normalization_797/Cast_2/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_99/batch_normalization_797/Cast_3/ReadVariableOpJgcn_attention_49/core_gcn_99/batch_normalization_797/Cast_3/ReadVariableOp2ђ
Tgcn_attention_49/core_gcn_99/batch_normalization_798/FusedBatchNormV3/ReadVariableOpTgcn_attention_49/core_gcn_99/batch_normalization_798/FusedBatchNormV3/ReadVariableOp2∞
Vgcn_attention_49/core_gcn_99/batch_normalization_798/FusedBatchNormV3/ReadVariableOp_1Vgcn_attention_49/core_gcn_99/batch_normalization_798/FusedBatchNormV3/ReadVariableOp_12К
Cgcn_attention_49/core_gcn_99/batch_normalization_798/ReadVariableOpCgcn_attention_49/core_gcn_99/batch_normalization_798/ReadVariableOp2О
Egcn_attention_49/core_gcn_99/batch_normalization_798/ReadVariableOp_1Egcn_attention_49/core_gcn_99/batch_normalization_798/ReadVariableOp_12Ф
Hgcn_attention_49/core_gcn_99/batch_normalization_799/Cast/ReadVariableOpHgcn_attention_49/core_gcn_99/batch_normalization_799/Cast/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_99/batch_normalization_799/Cast_1/ReadVariableOpJgcn_attention_49/core_gcn_99/batch_normalization_799/Cast_1/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_99/batch_normalization_799/Cast_2/ReadVariableOpJgcn_attention_49/core_gcn_99/batch_normalization_799/Cast_2/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_99/batch_normalization_799/Cast_3/ReadVariableOpJgcn_attention_49/core_gcn_99/batch_normalization_799/Cast_3/ReadVariableOp2А
>gcn_attention_49/core_gcn_99/conv1d_396/BiasAdd/ReadVariableOp>gcn_attention_49/core_gcn_99/conv1d_396/BiasAdd/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_99/conv1d_396/conv1d/ExpandDims_1/ReadVariableOpJgcn_attention_49/core_gcn_99/conv1d_396/conv1d/ExpandDims_1/ReadVariableOp2А
>gcn_attention_49/core_gcn_99/conv1d_397/BiasAdd/ReadVariableOp>gcn_attention_49/core_gcn_99/conv1d_397/BiasAdd/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_99/conv1d_397/conv1d/ExpandDims_1/ReadVariableOpJgcn_attention_49/core_gcn_99/conv1d_397/conv1d/ExpandDims_1/ReadVariableOp2А
>gcn_attention_49/core_gcn_99/conv1d_398/BiasAdd/ReadVariableOp>gcn_attention_49/core_gcn_99/conv1d_398/BiasAdd/ReadVariableOp2Ш
Jgcn_attention_49/core_gcn_99/conv1d_398/conv1d/ExpandDims_1/ReadVariableOpJgcn_attention_49/core_gcn_99/conv1d_398/conv1d/ExpandDims_1/ReadVariableOp2А
>gcn_attention_49/core_gcn_99/conv2d_199/BiasAdd/ReadVariableOp>gcn_attention_49/core_gcn_99/conv2d_199/BiasAdd/ReadVariableOp2~
=gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/ReadVariableOp=gcn_attention_49/core_gcn_99/conv2d_199/Conv2D/ReadVariableOp2h
2gcn_attention_49/dense_49/Tensordot/ReadVariableOp2gcn_attention_49/dense_49/Tensordot/ReadVariableOp2`
.gcn_attention_49/embedding_49/embedding_lookup.gcn_attention_49/embedding_49/embedding_lookup:b ^
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
%
_user_specified_nameinput_text1:nj
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
%
_user_specified_nameinput_text2:nj
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
%
_user_specified_nameinput_text3
р
Ж
S__inference_batch_normalization_789_layer_call_and_return_conditional_losses_611823

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identityИҐCast/ReadVariableOpҐCast_1/ReadVariableOpҐCast_2/ReadVariableOpҐCast_3/ReadVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityђ
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ*
‘
S__inference_batch_normalization_793_layer_call_and_return_conditional_losses_612495

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐCast/ReadVariableOpҐCast_1/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul…
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Г
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ*
‘
S__inference_batch_normalization_787_layer_call_and_return_conditional_losses_611595

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐCast/ReadVariableOpҐCast_1/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul…
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Г
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ*
‘
S__inference_batch_normalization_799_layer_call_and_return_conditional_losses_614676

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐCast/ReadVariableOpҐCast_1/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul…
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Г
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
е
Ю
S__inference_batch_normalization_786_layer_call_and_return_conditional_losses_611407

inputs%
readvariableop_resource:	'
readvariableop_1_resource:	6
(fusedbatchnormv3_readvariableop_resource:	8
*fusedbatchnormv3_readvariableop_1_resource:	
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:	*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:	*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	:	:	:	:	:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	2

IdentityЄ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	
 
_user_specified_nameinputs
ћ*
‘
S__inference_batch_normalization_796_layer_call_and_return_conditional_losses_612945

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐCast/ReadVariableOpҐCast_1/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul…
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Г
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
У	
”
8__inference_batch_normalization_784_layer_call_fn_613516

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCall≠
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_784_layer_call_and_return_conditional_losses_6112472
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
У	
”
8__inference_batch_normalization_791_layer_call_fn_614006

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCall≠
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_791_layer_call_and_return_conditional_losses_6121472
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
≈	
”
8__inference_batch_normalization_788_layer_call_fn_613797

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_788_layer_call_and_return_conditional_losses_6117392
StatefulPartitionedCallХ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ*
‘
S__inference_batch_normalization_787_layer_call_and_return_conditional_losses_613731

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐCast/ReadVariableOpҐCast_1/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul…
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Г
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
÷&
м
$__inference_signature_wrapper_611223
input_text1
input_text2
input_text3
unknown:ђ
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
identityИҐStatefulPartitionedCallу
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
 :€€€€€€€€€€€€€€€€€€*v
_read_only_resource_inputsX
VT	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUV*0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference_tf_translate_6110462
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Є
_input_shapes¶
£:€€€€€€€€€€€€€€€€€€ђ:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ
%
_user_specified_nameinput_text1:nj
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
%
_user_specified_nameinput_text2:nj
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
%
_user_specified_nameinput_text3
£
я
__inference_loss_fn_1_613753g
Mgcn_attention_49_conv2d_197_kernel_regularizer_square_readvariableop_resource:		
identityИҐDgcn_attention_49/conv2d_197/kernel/Regularizer/Square/ReadVariableOpҐ
Dgcn_attention_49/conv2d_197/kernel/Regularizer/Square/ReadVariableOpReadVariableOpMgcn_attention_49_conv2d_197_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:		*
dtype02F
Dgcn_attention_49/conv2d_197/kernel/Regularizer/Square/ReadVariableOpч
5gcn_attention_49/conv2d_197/kernel/Regularizer/SquareSquareLgcn_attention_49/conv2d_197/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		27
5gcn_attention_49/conv2d_197/kernel/Regularizer/Square≈
4gcn_attention_49/conv2d_197/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             26
4gcn_attention_49/conv2d_197/kernel/Regularizer/ConstК
2gcn_attention_49/conv2d_197/kernel/Regularizer/SumSum9gcn_attention_49/conv2d_197/kernel/Regularizer/Square:y:0=gcn_attention_49/conv2d_197/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 24
2gcn_attention_49/conv2d_197/kernel/Regularizer/Sum±
4gcn_attention_49/conv2d_197/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—826
4gcn_attention_49/conv2d_197/kernel/Regularizer/mul/xМ
2gcn_attention_49/conv2d_197/kernel/Regularizer/mulMul=gcn_attention_49/conv2d_197/kernel/Regularizer/mul/x:output:0;gcn_attention_49/conv2d_197/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 24
2gcn_attention_49/conv2d_197/kernel/Regularizer/mulА
IdentityIdentity6gcn_attention_49/conv2d_197/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityХ
NoOpNoOpE^gcn_attention_49/conv2d_197/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2М
Dgcn_attention_49/conv2d_197/kernel/Regularizer/Square/ReadVariableOpDgcn_attention_49/conv2d_197/kernel/Regularizer/Square/ReadVariableOp
«	
”
8__inference_batch_normalization_792_layer_call_fn_614086

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_792_layer_call_and_return_conditional_losses_6123072
StatefulPartitionedCallХ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
«	
”
8__inference_batch_normalization_788_layer_call_fn_613784

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_788_layer_call_and_return_conditional_losses_6116952
StatefulPartitionedCallХ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
С	
”
8__inference_batch_normalization_789_layer_call_fn_613859

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_789_layer_call_and_return_conditional_losses_6118832
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
р
Ж
S__inference_batch_normalization_791_layer_call_and_return_conditional_losses_612147

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identityИҐCast/ReadVariableOpҐCast_1/ReadVariableOpҐCast_2/ReadVariableOpҐCast_3/ReadVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityђ
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Щ
¬
S__inference_batch_normalization_792_layer_call_and_return_conditional_losses_612351

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3¬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueќ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity№
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ*
‘
S__inference_batch_normalization_791_layer_call_and_return_conditional_losses_612207

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐCast/ReadVariableOpҐCast_1/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul…
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Г
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ*
‘
S__inference_batch_normalization_784_layer_call_and_return_conditional_losses_613583

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐCast/ReadVariableOpҐCast_1/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul…
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Г
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
р
Ж
S__inference_batch_normalization_787_layer_call_and_return_conditional_losses_611535

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identityИҐCast/ReadVariableOpҐCast_1/ReadVariableOpҐCast_2/ReadVariableOpҐCast_3/ReadVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityђ
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
р
Ж
S__inference_batch_normalization_787_layer_call_and_return_conditional_losses_613697

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identityИҐCast/ReadVariableOpҐCast_1/ReadVariableOpҐCast_2/ReadVariableOpҐCast_3/ReadVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityђ
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Щ
¬
S__inference_batch_normalization_788_layer_call_and_return_conditional_losses_613833

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3¬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueќ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity№
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ*
‘
S__inference_batch_normalization_790_layer_call_and_return_conditional_losses_612045

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐCast/ReadVariableOpҐCast_1/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul…
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Г
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
√
ч
__inference_loss_fn_2_614226s
Ygcn_attention_49_core_gcn_98_conv2d_198_kernel_regularizer_square_readvariableop_resource:
identityИҐPgcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/Square/ReadVariableOp∆
Pgcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/Square/ReadVariableOpReadVariableOpYgcn_attention_49_core_gcn_98_conv2d_198_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype02R
Pgcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/Square/ReadVariableOpЫ
Agcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/SquareSquareXgcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2C
Agcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/SquareЁ
@gcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2B
@gcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/ConstЇ
>gcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/SumSumEgcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/Square:y:0Igcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2@
>gcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/Sum…
@gcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82B
@gcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/mul/xЉ
>gcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/mulMulIgcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/mul/x:output:0Ggcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2@
>gcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/mulМ
IdentityIdentityBgcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity°
NoOpNoOpQ^gcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2§
Pgcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/Square/ReadVariableOpPgcn_attention_49/core_gcn_98/conv2d_198/kernel/Regularizer/Square/ReadVariableOp
ћ*
‘
S__inference_batch_normalization_789_layer_call_and_return_conditional_losses_611883

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐCast/ReadVariableOpҐCast_1/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul…
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Г
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ*
‘
S__inference_batch_normalization_791_layer_call_and_return_conditional_losses_614073

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐCast/ReadVariableOpҐCast_1/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul…
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Г
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ*
‘
S__inference_batch_normalization_790_layer_call_and_return_conditional_losses_613993

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐCast/ReadVariableOpҐCast_1/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul…
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Г
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
С	
”
8__inference_batch_normalization_787_layer_call_fn_613677

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_787_layer_call_and_return_conditional_losses_6115952
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
У	
”
8__inference_batch_normalization_790_layer_call_fn_613926

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCall≠
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_790_layer_call_and_return_conditional_losses_6119852
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Щ
¬
S__inference_batch_normalization_794_layer_call_and_return_conditional_losses_614294

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3¬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueќ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity№
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ*
‘
S__inference_batch_normalization_797_layer_call_and_return_conditional_losses_614534

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐCast/ReadVariableOpҐCast_1/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul…
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Г
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
≈	
”
8__inference_batch_normalization_794_layer_call_fn_614258

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_794_layer_call_and_return_conditional_losses_6126392
StatefulPartitionedCallХ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
е
Ю
S__inference_batch_normalization_792_layer_call_and_return_conditional_losses_612307

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

IdentityЄ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
р
Ж
S__inference_batch_normalization_790_layer_call_and_return_conditional_losses_611985

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identityИҐCast/ReadVariableOpҐCast_1/ReadVariableOpҐCast_2/ReadVariableOpҐCast_3/ReadVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityђ
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
р
Ж
S__inference_batch_normalization_797_layer_call_and_return_conditional_losses_614500

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identityИҐCast/ReadVariableOpҐCast_1/ReadVariableOpҐCast_2/ReadVariableOpҐCast_3/ReadVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityђ
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
С	
”
8__inference_batch_normalization_790_layer_call_fn_613939

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_790_layer_call_and_return_conditional_losses_6120452
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
√
ч
__inference_loss_fn_3_614687s
Ygcn_attention_49_core_gcn_99_conv2d_199_kernel_regularizer_square_readvariableop_resource:
identityИҐPgcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/Square/ReadVariableOp∆
Pgcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/Square/ReadVariableOpReadVariableOpYgcn_attention_49_core_gcn_99_conv2d_199_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype02R
Pgcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/Square/ReadVariableOpЫ
Agcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/SquareSquareXgcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2C
Agcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/SquareЁ
@gcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2B
@gcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/ConstЇ
>gcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/SumSumEgcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/Square:y:0Igcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2@
>gcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/Sum…
@gcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ј—82B
@gcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/mul/xЉ
>gcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/mulMulIgcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/mul/x:output:0Ggcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2@
>gcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/mulМ
IdentityIdentityBgcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity°
NoOpNoOpQ^gcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2§
Pgcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/Square/ReadVariableOpPgcn_attention_49/core_gcn_99/conv2d_199/kernel/Regularizer/Square/ReadVariableOp
С	
”
8__inference_batch_normalization_784_layer_call_fn_613529

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_784_layer_call_and_return_conditional_losses_6113072
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
р
Ж
S__inference_batch_normalization_797_layer_call_and_return_conditional_losses_613047

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identityИҐCast/ReadVariableOpҐCast_1/ReadVariableOpҐCast_2/ReadVariableOpҐCast_3/ReadVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityђ
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
р
Ж
S__inference_batch_normalization_796_layer_call_and_return_conditional_losses_614420

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identityИҐCast/ReadVariableOpҐCast_1/ReadVariableOpҐCast_2/ReadVariableOpҐCast_3/ReadVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityђ
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
С	
”
8__inference_batch_normalization_791_layer_call_fn_614019

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_791_layer_call_and_return_conditional_losses_6122072
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
р
Ж
S__inference_batch_normalization_793_layer_call_and_return_conditional_losses_614181

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identityИҐCast/ReadVariableOpҐCast_1/ReadVariableOpҐCast_2/ReadVariableOpҐCast_3/ReadVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityђ
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
У	
”
8__inference_batch_normalization_793_layer_call_fn_614148

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCall≠
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_793_layer_call_and_return_conditional_losses_6124352
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
р
Ж
S__inference_batch_normalization_799_layer_call_and_return_conditional_losses_614642

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identityИҐCast/ReadVariableOpҐCast_1/ReadVariableOpҐCast_2/ReadVariableOpҐCast_3/ReadVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityђ
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
р
Ж
S__inference_batch_normalization_790_layer_call_and_return_conditional_losses_613959

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identityИҐCast/ReadVariableOpҐCast_1/ReadVariableOpҐCast_2/ReadVariableOpҐCast_3/ReadVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityђ
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
е
Ю
S__inference_batch_normalization_794_layer_call_and_return_conditional_losses_614276

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

IdentityЄ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
р
Ж
S__inference_batch_normalization_791_layer_call_and_return_conditional_losses_614039

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identityИҐCast/ReadVariableOpҐCast_1/ReadVariableOpҐCast_2/ReadVariableOpҐCast_3/ReadVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityђ
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ*
‘
S__inference_batch_normalization_793_layer_call_and_return_conditional_losses_614215

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐCast/ReadVariableOpҐCast_1/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul…
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Г
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
е
Ю
S__inference_batch_normalization_792_layer_call_and_return_conditional_losses_614117

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

IdentityЄ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
р
Ж
S__inference_batch_normalization_784_layer_call_and_return_conditional_losses_611247

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identityИҐCast/ReadVariableOpҐCast_1/ReadVariableOpҐCast_2/ReadVariableOpҐCast_3/ReadVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityђ
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ*
‘
S__inference_batch_normalization_784_layer_call_and_return_conditional_losses_611307

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐCast/ReadVariableOpҐCast_1/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul…
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Г
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
р
Ж
S__inference_batch_normalization_784_layer_call_and_return_conditional_losses_613549

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identityИҐCast/ReadVariableOpҐCast_1/ReadVariableOpҐCast_2/ReadVariableOpҐCast_3/ReadVariableOpГ
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpЙ
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yЕ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityђ
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ*
‘
S__inference_batch_normalization_795_layer_call_and_return_conditional_losses_614374

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐCast/ReadVariableOpҐCast_1/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
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
„#<2
AssignMovingAvg/decay§
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mulњ
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
„#<2
AssignMovingAvg_1/decay™
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp†
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul…
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Г
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOpЙ
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
 *oГ:2
batchnorm/add/yВ
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
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
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
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2

Identityж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Щ
¬
S__inference_batch_normalization_786_layer_call_and_return_conditional_losses_613651

inputs%
readvariableop_resource:	'
readvariableop_1_resource:	6
(fusedbatchnormv3_readvariableop_resource:	8
*fusedbatchnormv3_readvariableop_1_resource:	
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:	*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:	*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	:	:	:	:	:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3¬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueќ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	2

Identity№
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€	
 
_user_specified_nameinputs
е
Ю
S__inference_batch_normalization_794_layer_call_and_return_conditional_losses_612595

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

IdentityЄ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs"®L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Л
serving_defaultч
Q
input_text1B
serving_default_input_text1:0€€€€€€€€€€€€€€€€€€ђ
]
input_text2N
serving_default_input_text2:0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
]
input_text3N
serving_default_input_text3:0+€€€€€€€€€€€€€€€€€€€€€€€€€€€H
outputs=
StatefulPartitionedCall:0€€€€€€€€€€€€€€€€€€tensorflow/serving/predict:ъХ
[
self_attention_layer

signatures
Љtf_translate"
_generic_user_object
щ
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
trainable_variables
regularization_losses
	variables
	keras_api
љ__call__
+Њ&call_and_return_all_conditional_losses"
_tf_keras_layer
-
њserving_default"
signature_map
љ

kernel
bias
trainable_variables
regularization_losses
 	variables
!	keras_api
ј__call__
+Ѕ&call_and_return_all_conditional_losses"
_tf_keras_layer
І
"trainable_variables
#regularization_losses
$	variables
%	keras_api
¬__call__
+√&call_and_return_all_conditional_losses"
_tf_keras_layer
м
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+trainable_variables
,regularization_losses
-	variables
.	keras_api
ƒ__call__
+≈&call_and_return_all_conditional_losses"
_tf_keras_layer
І
/trainable_variables
0regularization_losses
1	variables
2	keras_api
∆__call__
+«&call_and_return_all_conditional_losses"
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
≥

7kernel
8trainable_variables
9regularization_losses
:	variables
;	keras_api
»__call__
+…&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

<kernel
=bias
>trainable_variables
?regularization_losses
@	variables
A	keras_api
 __call__
+Ћ&call_and_return_all_conditional_losses"
_tf_keras_layer
І
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
ћ__call__
+Ќ&call_and_return_all_conditional_losses"
_tf_keras_layer
м
Faxis
	Ggamma
Hbeta
Imoving_mean
Jmoving_variance
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
ќ__call__
+ѕ&call_and_return_all_conditional_losses"
_tf_keras_layer
І
Otrainable_variables
Pregularization_losses
Q	variables
R	keras_api
–__call__
+—&call_and_return_all_conditional_losses"
_tf_keras_layer
Ј
S
embeddings
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
“__call__
+”&call_and_return_all_conditional_losses"
_tf_keras_layer
І
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
‘__call__
+’&call_and_return_all_conditional_losses"
_tf_keras_layer
м
\axis
	]gamma
^beta
_moving_mean
`moving_variance
atrainable_variables
bregularization_losses
c	variables
d	keras_api
÷__call__
+„&call_and_return_all_conditional_losses"
_tf_keras_layer
І
etrainable_variables
fregularization_losses
g	variables
h	keras_api
Ў__call__
+ў&call_and_return_all_conditional_losses"
_tf_keras_layer
.
i0
j1"
trackable_list_wrapper
(
k	keras_api"
_tf_keras_layer
љ

lkernel
mbias
ntrainable_variables
oregularization_losses
p	variables
q	keras_api
Џ__call__
+џ&call_and_return_all_conditional_losses"
_tf_keras_layer
І
rtrainable_variables
sregularization_losses
t	variables
u	keras_api
№__call__
+Ё&call_and_return_all_conditional_losses"
_tf_keras_layer
д
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
А22
Б23
В24
Г25
Д26
Е27
Ж28
З29
И30
Й31
К32
Л33
М34
Н35
О36
П37
Р38
С39
Т40
У41
Ф42
Х43
Ц44
Ч45
Ш46
Щ47
Ъ48
Ы49
Ь50
Э51
l52
m53"
trackable_list_wrapper
0
ё0
я1"
trackable_list_wrapper
м
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
А22
Б23
В24
Г25
Д26
Е27
Ж28
З29
И30
Й31
К32
Л33
М34
Н35
О36
П37
Р38
С39
Т40
У41
Ф42
Х43
Ц44
Ч45
Ш46
Щ47
Ъ48
Ы49
Ь50
Э51
l52
m53
)54
*55
I56
J57
_58
`59
Ю60
Я61
†62
°63
Ґ64
£65
§66
•67
¶68
І69
®70
©71
™72
Ђ73
ђ74
≠75
Ѓ76
ѓ77
∞78
±79
≤80
≥81
і82
µ83"
trackable_list_wrapper
µ
 ґlayer_regularization_losses
trainable_variables
regularization_losses
Јmetrics
	variables
Єlayers
єnon_trainable_variables
Їlayer_metrics
љ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
9:7ђ2"gcn_attention_49/conv1d_392/kernel
.:,2 gcn_attention_49/conv1d_392/bias
.
0
1"
trackable_list_wrapper
(
ё0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
µ
 їlayer_regularization_losses
trainable_variables
regularization_losses
Љmetrics
 	variables
љlayers
Њnon_trainable_variables
њlayer_metrics
ј__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 јlayer_regularization_losses
"trainable_variables
#regularization_losses
Ѕmetrics
$	variables
¬layers
√non_trainable_variables
ƒlayer_metrics
¬__call__
+√&call_and_return_all_conditional_losses
'√"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<::2.gcn_attention_49/batch_normalization_784/gamma
;:92-gcn_attention_49/batch_normalization_784/beta
D:B (24gcn_attention_49/batch_normalization_784/moving_mean
H:F (28gcn_attention_49/batch_normalization_784/moving_variance
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
'0
(1
)2
*3"
trackable_list_wrapper
µ
 ≈layer_regularization_losses
+trainable_variables
,regularization_losses
∆metrics
-	variables
«layers
»non_trainable_variables
…layer_metrics
ƒ__call__
+≈&call_and_return_all_conditional_losses
'≈"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
  layer_regularization_losses
/trainable_variables
0regularization_losses
Ћmetrics
1	variables
ћlayers
Ќnon_trainable_variables
ќlayer_metrics
∆__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
2:0	2 gcn_attention_49/dense_49/kernel
'
70"
trackable_list_wrapper
 "
trackable_list_wrapper
'
70"
trackable_list_wrapper
µ
 ѕlayer_regularization_losses
8trainable_variables
9regularization_losses
–metrics
:	variables
—layers
“non_trainable_variables
”layer_metrics
»__call__
+…&call_and_return_all_conditional_losses
'…"call_and_return_conditional_losses"
_generic_user_object
<::		2"gcn_attention_49/conv2d_197/kernel
.:,	2 gcn_attention_49/conv2d_197/bias
.
<0
=1"
trackable_list_wrapper
(
я0"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
µ
 ‘layer_regularization_losses
>trainable_variables
?regularization_losses
’metrics
@	variables
÷layers
„non_trainable_variables
Ўlayer_metrics
 __call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ўlayer_regularization_losses
Btrainable_variables
Cregularization_losses
Џmetrics
D	variables
џlayers
№non_trainable_variables
Ёlayer_metrics
ћ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<::	2.gcn_attention_49/batch_normalization_786/gamma
;:9	2-gcn_attention_49/batch_normalization_786/beta
D:B	 (24gcn_attention_49/batch_normalization_786/moving_mean
H:F	 (28gcn_attention_49/batch_normalization_786/moving_variance
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
G0
H1
I2
J3"
trackable_list_wrapper
µ
 ёlayer_regularization_losses
Ktrainable_variables
Lregularization_losses
яmetrics
M	variables
аlayers
бnon_trainable_variables
вlayer_metrics
ќ__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 гlayer_regularization_losses
Otrainable_variables
Pregularization_losses
дmetrics
Q	variables
еlayers
жnon_trainable_variables
зlayer_metrics
–__call__
+—&call_and_return_all_conditional_losses
'—"call_and_return_conditional_losses"
_generic_user_object
::8	2(gcn_attention_49/embedding_49/embeddings
'
S0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
S0"
trackable_list_wrapper
µ
 иlayer_regularization_losses
Ttrainable_variables
Uregularization_losses
йmetrics
V	variables
кlayers
лnon_trainable_variables
мlayer_metrics
“__call__
+”&call_and_return_all_conditional_losses
'”"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 нlayer_regularization_losses
Xtrainable_variables
Yregularization_losses
оmetrics
Z	variables
пlayers
рnon_trainable_variables
сlayer_metrics
‘__call__
+’&call_and_return_all_conditional_losses
'’"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<::2.gcn_attention_49/batch_normalization_787/gamma
;:92-gcn_attention_49/batch_normalization_787/beta
D:B (24gcn_attention_49/batch_normalization_787/moving_mean
H:F (28gcn_attention_49/batch_normalization_787/moving_variance
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
]0
^1
_2
`3"
trackable_list_wrapper
µ
 тlayer_regularization_losses
atrainable_variables
bregularization_losses
уmetrics
c	variables
фlayers
хnon_trainable_variables
цlayer_metrics
÷__call__
+„&call_and_return_all_conditional_losses
'„"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 чlayer_regularization_losses
etrainable_variables
fregularization_losses
шmetrics
g	variables
щlayers
ъnon_trainable_variables
ыlayer_metrics
Ў__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
Щ
ьconv2d_1
эbatch_norm_1
ю	dropout_1
€conv1d_1
Аbatch_norm_2
Б	dropout_2
Вconv1d_2
Гbatch_norm_3
Д	dropout_3
Еconv1d_3
Жbatch_norm_4
З	dropout_4
Иbatch_norm_5
Йbatch_norm_6
Кtrainable_variables
Лregularization_losses
М	variables
Н	keras_api
а__call__
+б&call_and_return_all_conditional_losses"
_tf_keras_layer
Щ
Оconv2d_1
Пbatch_norm_1
Р	dropout_1
Сconv1d_1
Тbatch_norm_2
У	dropout_2
Фconv1d_2
Хbatch_norm_3
Ц	dropout_3
Чconv1d_3
Шbatch_norm_4
Щ	dropout_4
Ъbatch_norm_5
Ыbatch_norm_6
Ьtrainable_variables
Эregularization_losses
Ю	variables
Я	keras_api
в__call__
+г&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_generic_user_object
8:62"gcn_attention_49/conv1d_399/kernel
.:,2 gcn_attention_49/conv1d_399/bias
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
µ
 †layer_regularization_losses
ntrainable_variables
oregularization_losses
°metrics
p	variables
Ґlayers
£non_trainable_variables
§layer_metrics
Џ__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 •layer_regularization_losses
rtrainable_variables
sregularization_losses
¶metrics
t	variables
Іlayers
®non_trainable_variables
©layer_metrics
№__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
H:F2.gcn_attention_49/core_gcn_98/conv2d_198/kernel
::82,gcn_attention_49/core_gcn_98/conv2d_198/bias
H:F2:gcn_attention_49/core_gcn_98/batch_normalization_788/gamma
G:E29gcn_attention_49/core_gcn_98/batch_normalization_788/beta
D:B2.gcn_attention_49/core_gcn_98/conv1d_393/kernel
::82,gcn_attention_49/core_gcn_98/conv1d_393/bias
H:F2:gcn_attention_49/core_gcn_98/batch_normalization_789/gamma
G:E29gcn_attention_49/core_gcn_98/batch_normalization_789/beta
D:B2.gcn_attention_49/core_gcn_98/conv1d_394/kernel
::82,gcn_attention_49/core_gcn_98/conv1d_394/bias
H:F2:gcn_attention_49/core_gcn_98/batch_normalization_790/gamma
G:E29gcn_attention_49/core_gcn_98/batch_normalization_790/beta
D:B2.gcn_attention_49/core_gcn_98/conv1d_395/kernel
::82,gcn_attention_49/core_gcn_98/conv1d_395/bias
H:F2:gcn_attention_49/core_gcn_98/batch_normalization_791/gamma
G:E29gcn_attention_49/core_gcn_98/batch_normalization_791/beta
H:F2:gcn_attention_49/core_gcn_98/batch_normalization_792/gamma
G:E29gcn_attention_49/core_gcn_98/batch_normalization_792/beta
H:F2:gcn_attention_49/core_gcn_98/batch_normalization_793/gamma
G:E29gcn_attention_49/core_gcn_98/batch_normalization_793/beta
H:F2.gcn_attention_49/core_gcn_99/conv2d_199/kernel
::82,gcn_attention_49/core_gcn_99/conv2d_199/bias
H:F2:gcn_attention_49/core_gcn_99/batch_normalization_794/gamma
G:E29gcn_attention_49/core_gcn_99/batch_normalization_794/beta
D:B2.gcn_attention_49/core_gcn_99/conv1d_396/kernel
::82,gcn_attention_49/core_gcn_99/conv1d_396/bias
H:F2:gcn_attention_49/core_gcn_99/batch_normalization_795/gamma
G:E29gcn_attention_49/core_gcn_99/batch_normalization_795/beta
D:B2.gcn_attention_49/core_gcn_99/conv1d_397/kernel
::82,gcn_attention_49/core_gcn_99/conv1d_397/bias
H:F2:gcn_attention_49/core_gcn_99/batch_normalization_796/gamma
G:E29gcn_attention_49/core_gcn_99/batch_normalization_796/beta
D:B2.gcn_attention_49/core_gcn_99/conv1d_398/kernel
::82,gcn_attention_49/core_gcn_99/conv1d_398/bias
H:F2:gcn_attention_49/core_gcn_99/batch_normalization_797/gamma
G:E29gcn_attention_49/core_gcn_99/batch_normalization_797/beta
H:F2:gcn_attention_49/core_gcn_99/batch_normalization_798/gamma
G:E29gcn_attention_49/core_gcn_99/batch_normalization_798/beta
H:F2:gcn_attention_49/core_gcn_99/batch_normalization_799/gamma
G:E29gcn_attention_49/core_gcn_99/batch_normalization_799/beta
P:N (2@gcn_attention_49/core_gcn_98/batch_normalization_788/moving_mean
T:R (2Dgcn_attention_49/core_gcn_98/batch_normalization_788/moving_variance
P:N (2@gcn_attention_49/core_gcn_98/batch_normalization_789/moving_mean
T:R (2Dgcn_attention_49/core_gcn_98/batch_normalization_789/moving_variance
P:N (2@gcn_attention_49/core_gcn_98/batch_normalization_790/moving_mean
T:R (2Dgcn_attention_49/core_gcn_98/batch_normalization_790/moving_variance
P:N (2@gcn_attention_49/core_gcn_98/batch_normalization_791/moving_mean
T:R (2Dgcn_attention_49/core_gcn_98/batch_normalization_791/moving_variance
P:N (2@gcn_attention_49/core_gcn_98/batch_normalization_792/moving_mean
T:R (2Dgcn_attention_49/core_gcn_98/batch_normalization_792/moving_variance
P:N (2@gcn_attention_49/core_gcn_98/batch_normalization_793/moving_mean
T:R (2Dgcn_attention_49/core_gcn_98/batch_normalization_793/moving_variance
P:N (2@gcn_attention_49/core_gcn_99/batch_normalization_794/moving_mean
T:R (2Dgcn_attention_49/core_gcn_99/batch_normalization_794/moving_variance
P:N (2@gcn_attention_49/core_gcn_99/batch_normalization_795/moving_mean
T:R (2Dgcn_attention_49/core_gcn_99/batch_normalization_795/moving_variance
P:N (2@gcn_attention_49/core_gcn_99/batch_normalization_796/moving_mean
T:R (2Dgcn_attention_49/core_gcn_99/batch_normalization_796/moving_variance
P:N (2@gcn_attention_49/core_gcn_99/batch_normalization_797/moving_mean
T:R (2Dgcn_attention_49/core_gcn_99/batch_normalization_797/moving_variance
P:N (2@gcn_attention_49/core_gcn_99/batch_normalization_798/moving_mean
T:R (2Dgcn_attention_49/core_gcn_99/batch_normalization_798/moving_variance
P:N (2@gcn_attention_49/core_gcn_99/batch_normalization_799/moving_mean
T:R (2Dgcn_attention_49/core_gcn_99/batch_normalization_799/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∆
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
Ю
)0
*1
I2
J3
_4
`5
Ю6
Я7
†8
°9
Ґ10
£11
§12
•13
¶14
І15
®16
©17
™18
Ђ19
ђ20
≠21
Ѓ22
ѓ23
∞24
±25
≤26
≥27
і28
µ29"
trackable_list_wrapper
 "
trackable_dict_wrapper
(
ё0"
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
.
)0
*1"
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
я0"
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
.
I0
J1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
_0
`1"
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
Ѕ

vkernel
wbias
™trainable_variables
Ђregularization_losses
ђ	variables
≠	keras_api
д__call__
+е&call_and_return_all_conditional_losses"
_tf_keras_layer
у
	Ѓaxis
	xgamma
ybeta
Юmoving_mean
Яmoving_variance
ѓtrainable_variables
∞regularization_losses
±	variables
≤	keras_api
ж__call__
+з&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
≥trainable_variables
іregularization_losses
µ	variables
ґ	keras_api
и__call__
+й&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ

zkernel
{bias
Јtrainable_variables
Єregularization_losses
є	variables
Ї	keras_api
к__call__
+л&call_and_return_all_conditional_losses"
_tf_keras_layer
у
	їaxis
	|gamma
}beta
†moving_mean
°moving_variance
Љtrainable_variables
љregularization_losses
Њ	variables
њ	keras_api
м__call__
+н&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
јtrainable_variables
Ѕregularization_losses
¬	variables
√	keras_api
о__call__
+п&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ

~kernel
bias
ƒtrainable_variables
≈regularization_losses
∆	variables
«	keras_api
р__call__
+с&call_and_return_all_conditional_losses"
_tf_keras_layer
х
	»axis

Аgamma
	Бbeta
Ґmoving_mean
£moving_variance
…trainable_variables
 regularization_losses
Ћ	variables
ћ	keras_api
т__call__
+у&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Ќtrainable_variables
ќregularization_losses
ѕ	variables
–	keras_api
ф__call__
+х&call_and_return_all_conditional_losses"
_tf_keras_layer
√
Вkernel
	Гbias
—trainable_variables
“regularization_losses
”	variables
‘	keras_api
ц__call__
+ч&call_and_return_all_conditional_losses"
_tf_keras_layer
х
	’axis

Дgamma
	Еbeta
§moving_mean
•moving_variance
÷trainable_variables
„regularization_losses
Ў	variables
ў	keras_api
ш__call__
+щ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Џtrainable_variables
џregularization_losses
№	variables
Ё	keras_api
ъ__call__
+ы&call_and_return_all_conditional_losses"
_tf_keras_layer
х
	ёaxis

Жgamma
	Зbeta
¶moving_mean
Іmoving_variance
яtrainable_variables
аregularization_losses
б	variables
в	keras_api
ь__call__
+э&call_and_return_all_conditional_losses"
_tf_keras_layer
х
	гaxis

Иgamma
	Йbeta
®moving_mean
©moving_variance
дtrainable_variables
еregularization_losses
ж	variables
з	keras_api
ю__call__
+€&call_and_return_all_conditional_losses"
_tf_keras_layer
ј
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
А10
Б11
В12
Г13
Д14
Е15
Ж16
З17
И18
Й19"
trackable_list_wrapper
(
А0"
trackable_list_wrapper
ђ
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
А10
Б11
В12
Г13
Д14
Е15
Ж16
З17
И18
Й19
Ю20
Я21
†22
°23
Ґ24
£25
§26
•27
¶28
І29
®30
©31"
trackable_list_wrapper
Є
 иlayer_regularization_losses
Кtrainable_variables
Лregularization_losses
йmetrics
М	variables
кlayers
лnon_trainable_variables
мlayer_metrics
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
√
Кkernel
	Лbias
нtrainable_variables
оregularization_losses
п	variables
р	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"
_tf_keras_layer
х
	сaxis

Мgamma
	Нbeta
™moving_mean
Ђmoving_variance
тtrainable_variables
уregularization_losses
ф	variables
х	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
цtrainable_variables
чregularization_losses
ш	variables
щ	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"
_tf_keras_layer
√
Оkernel
	Пbias
ъtrainable_variables
ыregularization_losses
ь	variables
э	keras_api
З__call__
+И&call_and_return_all_conditional_losses"
_tf_keras_layer
х
	юaxis

Рgamma
	Сbeta
ђmoving_mean
≠moving_variance
€trainable_variables
Аregularization_losses
Б	variables
В	keras_api
Й__call__
+К&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Гtrainable_variables
Дregularization_losses
Е	variables
Ж	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"
_tf_keras_layer
√
Тkernel
	Уbias
Зtrainable_variables
Иregularization_losses
Й	variables
К	keras_api
Н__call__
+О&call_and_return_all_conditional_losses"
_tf_keras_layer
х
	Лaxis

Фgamma
	Хbeta
Ѓmoving_mean
ѓmoving_variance
Мtrainable_variables
Нregularization_losses
О	variables
П	keras_api
П__call__
+Р&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Рtrainable_variables
Сregularization_losses
Т	variables
У	keras_api
С__call__
+Т&call_and_return_all_conditional_losses"
_tf_keras_layer
√
Цkernel
	Чbias
Фtrainable_variables
Хregularization_losses
Ц	variables
Ч	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses"
_tf_keras_layer
х
	Шaxis

Шgamma
	Щbeta
∞moving_mean
±moving_variance
Щtrainable_variables
Ъregularization_losses
Ы	variables
Ь	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Эtrainable_variables
Юregularization_losses
Я	variables
†	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses"
_tf_keras_layer
х
	°axis

Ъgamma
	Ыbeta
≤moving_mean
≥moving_variance
Ґtrainable_variables
£regularization_losses
§	variables
•	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses"
_tf_keras_layer
х
	¶axis

Ьgamma
	Эbeta
іmoving_mean
µmoving_variance
Іtrainable_variables
®regularization_losses
©	variables
™	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses"
_tf_keras_layer
 
К0
Л1
М2
Н3
О4
П5
Р6
С7
Т8
У9
Ф10
Х11
Ц12
Ч13
Ш14
Щ15
Ъ16
Ы17
Ь18
Э19"
trackable_list_wrapper
(
Э0"
trackable_list_wrapper
ґ
К0
Л1
М2
Н3
О4
П5
Р6
С7
Т8
У9
Ф10
Х11
Ц12
Ч13
Ш14
Щ15
Ъ16
Ы17
Ь18
Э19
™20
Ђ21
ђ22
≠23
Ѓ24
ѓ25
∞26
±27
≤28
≥29
і30
µ31"
trackable_list_wrapper
Є
 Ђlayer_regularization_losses
Ьtrainable_variables
Эregularization_losses
ђmetrics
Ю	variables
≠layers
Ѓnon_trainable_variables
ѓlayer_metrics
в__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
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
.
v0
w1"
trackable_list_wrapper
(
А0"
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
Є
 ∞layer_regularization_losses
™trainable_variables
Ђregularization_losses
±metrics
ђ	variables
≤layers
≥non_trainable_variables
іlayer_metrics
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
>
x0
y1
Ю2
Я3"
trackable_list_wrapper
Є
 µlayer_regularization_losses
ѓtrainable_variables
∞regularization_losses
ґmetrics
±	variables
Јlayers
Єnon_trainable_variables
єlayer_metrics
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 Їlayer_regularization_losses
≥trainable_variables
іregularization_losses
їmetrics
µ	variables
Љlayers
љnon_trainable_variables
Њlayer_metrics
и__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
Є
 њlayer_regularization_losses
Јtrainable_variables
Єregularization_losses
јmetrics
є	variables
Ѕlayers
¬non_trainable_variables
√layer_metrics
к__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
>
|0
}1
†2
°3"
trackable_list_wrapper
Є
 ƒlayer_regularization_losses
Љtrainable_variables
љregularization_losses
≈metrics
Њ	variables
∆layers
«non_trainable_variables
»layer_metrics
м__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 …layer_regularization_losses
јtrainable_variables
Ѕregularization_losses
 metrics
¬	variables
Ћlayers
ћnon_trainable_variables
Ќlayer_metrics
о__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
Є
 ќlayer_regularization_losses
ƒtrainable_variables
≈regularization_losses
ѕmetrics
∆	variables
–layers
—non_trainable_variables
“layer_metrics
р__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
А0
Б1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
А0
Б1
Ґ2
£3"
trackable_list_wrapper
Є
 ”layer_regularization_losses
…trainable_variables
 regularization_losses
‘metrics
Ћ	variables
’layers
÷non_trainable_variables
„layer_metrics
т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 Ўlayer_regularization_losses
Ќtrainable_variables
ќregularization_losses
ўmetrics
ѕ	variables
Џlayers
џnon_trainable_variables
№layer_metrics
ф__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
0
В0
Г1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
В0
Г1"
trackable_list_wrapper
Є
 Ёlayer_regularization_losses
—trainable_variables
“regularization_losses
ёmetrics
”	variables
яlayers
аnon_trainable_variables
бlayer_metrics
ц__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
Д0
Е1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
Д0
Е1
§2
•3"
trackable_list_wrapper
Є
 вlayer_regularization_losses
÷trainable_variables
„regularization_losses
гmetrics
Ў	variables
дlayers
еnon_trainable_variables
жlayer_metrics
ш__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 зlayer_regularization_losses
Џtrainable_variables
џregularization_losses
иmetrics
№	variables
йlayers
кnon_trainable_variables
лlayer_metrics
ъ__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
Ж0
З1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
Ж0
З1
¶2
І3"
trackable_list_wrapper
Є
 мlayer_regularization_losses
яtrainable_variables
аregularization_losses
нmetrics
б	variables
оlayers
пnon_trainable_variables
рlayer_metrics
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
И0
Й1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
И0
Й1
®2
©3"
trackable_list_wrapper
Є
 сlayer_regularization_losses
дtrainable_variables
еregularization_losses
тmetrics
ж	variables
уlayers
фnon_trainable_variables
хlayer_metrics
ю__call__
+€&call_and_return_all_conditional_losses
'€"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ф
ь0
э1
ю2
€3
А4
Б5
В6
Г7
Д8
Е9
Ж10
З11
И12
Й13"
trackable_list_wrapper
В
Ю0
Я1
†2
°3
Ґ4
£5
§6
•7
¶8
І9
®10
©11"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
К0
Л1"
trackable_list_wrapper
(
Э0"
trackable_list_wrapper
0
К0
Л1"
trackable_list_wrapper
Є
 цlayer_regularization_losses
нtrainable_variables
оregularization_losses
чmetrics
п	variables
шlayers
щnon_trainable_variables
ъlayer_metrics
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
М0
Н1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
М0
Н1
™2
Ђ3"
trackable_list_wrapper
Є
 ыlayer_regularization_losses
тtrainable_variables
уregularization_losses
ьmetrics
ф	variables
эlayers
юnon_trainable_variables
€layer_metrics
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 Аlayer_regularization_losses
цtrainable_variables
чregularization_losses
Бmetrics
ш	variables
Вlayers
Гnon_trainable_variables
Дlayer_metrics
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
0
О0
П1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
О0
П1"
trackable_list_wrapper
Є
 Еlayer_regularization_losses
ъtrainable_variables
ыregularization_losses
Жmetrics
ь	variables
Зlayers
Иnon_trainable_variables
Йlayer_metrics
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
Р0
С1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
Р0
С1
ђ2
≠3"
trackable_list_wrapper
Є
 Кlayer_regularization_losses
€trainable_variables
Аregularization_losses
Лmetrics
Б	variables
Мlayers
Нnon_trainable_variables
Оlayer_metrics
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 Пlayer_regularization_losses
Гtrainable_variables
Дregularization_losses
Рmetrics
Е	variables
Сlayers
Тnon_trainable_variables
Уlayer_metrics
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
0
Т0
У1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Т0
У1"
trackable_list_wrapper
Є
 Фlayer_regularization_losses
Зtrainable_variables
Иregularization_losses
Хmetrics
Й	variables
Цlayers
Чnon_trainable_variables
Шlayer_metrics
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
Ф0
Х1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
Ф0
Х1
Ѓ2
ѓ3"
trackable_list_wrapper
Є
 Щlayer_regularization_losses
Мtrainable_variables
Нregularization_losses
Ъmetrics
О	variables
Ыlayers
Ьnon_trainable_variables
Эlayer_metrics
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 Юlayer_regularization_losses
Рtrainable_variables
Сregularization_losses
Яmetrics
Т	variables
†layers
°non_trainable_variables
Ґlayer_metrics
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
0
Ц0
Ч1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ц0
Ч1"
trackable_list_wrapper
Є
 £layer_regularization_losses
Фtrainable_variables
Хregularization_losses
§metrics
Ц	variables
•layers
¶non_trainable_variables
Іlayer_metrics
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
Ш0
Щ1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
Ш0
Щ1
∞2
±3"
trackable_list_wrapper
Є
 ®layer_regularization_losses
Щtrainable_variables
Ъregularization_losses
©metrics
Ы	variables
™layers
Ђnon_trainable_variables
ђlayer_metrics
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
 ≠layer_regularization_losses
Эtrainable_variables
Юregularization_losses
Ѓmetrics
Я	variables
ѓlayers
∞non_trainable_variables
±layer_metrics
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
Ъ0
Ы1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
Ъ0
Ы1
≤2
≥3"
trackable_list_wrapper
Є
 ≤layer_regularization_losses
Ґtrainable_variables
£regularization_losses
≥metrics
§	variables
іlayers
µnon_trainable_variables
ґlayer_metrics
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
Ь0
Э1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
Ь0
Э1
і2
µ3"
trackable_list_wrapper
Є
 Јlayer_regularization_losses
Іtrainable_variables
®regularization_losses
Єmetrics
©	variables
єlayers
Їnon_trainable_variables
їlayer_metrics
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ф
О0
П1
Р2
С3
Т4
У5
Ф6
Х7
Ц8
Ч9
Ш10
Щ11
Ъ12
Ы13"
trackable_list_wrapper
В
™0
Ђ1
ђ2
≠3
Ѓ4
ѓ5
∞6
±7
≤8
≥9
і10
µ11"
trackable_list_wrapper
 "
trackable_dict_wrapper
(
А0"
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
0
Ю0
Я1"
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
0
†0
°1"
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
0
Ґ0
£1"
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
0
§0
•1"
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
0
¶0
І1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
®0
©1"
trackable_list_wrapper
 "
trackable_dict_wrapper
(
Э0"
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
0
™0
Ђ1"
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
0
ђ0
≠1"
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
0
Ѓ0
ѓ1"
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
0
∞0
±1"
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
0
≤0
≥1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
і0
µ1"
trackable_list_wrapper
 "
trackable_dict_wrapper
€2ь
__inference_tf_translate_611046Ў
Љ≤Є
FullArgSpec@
args8Ъ5
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

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *ФҐР
&К#€€€€€€€€€€€€€€€€€€ђ
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
ф2со
е≤б
FullArgSpecd
args\ЪY
jself
jbatch_target_feat
jseq_contact_batch
jpdb_distance_pair_batch

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ф2со
е≤б
FullArgSpecd
args\ЪY
jself
jbatch_target_feat
jseq_contact_batch
jpdb_distance_pair_batch

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
зBд
$__inference_signature_wrapper_611223input_text1input_text2input_text3"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ѓ2Ђ
8__inference_batch_normalization_784_layer_call_fn_613516
8__inference_batch_normalization_784_layer_call_fn_613529і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
д2б
S__inference_batch_normalization_784_layer_call_and_return_conditional_losses_613549
S__inference_batch_normalization_784_layer_call_and_return_conditional_losses_613583і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ѓ2Ђ
8__inference_batch_normalization_786_layer_call_fn_613602
8__inference_batch_normalization_786_layer_call_fn_613615і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
д2б
S__inference_batch_normalization_786_layer_call_and_return_conditional_losses_613633
S__inference_batch_normalization_786_layer_call_and_return_conditional_losses_613651і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ѓ2Ђ
8__inference_batch_normalization_787_layer_call_fn_613664
8__inference_batch_normalization_787_layer_call_fn_613677і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
д2б
S__inference_batch_normalization_787_layer_call_and_return_conditional_losses_613697
S__inference_batch_normalization_787_layer_call_and_return_conditional_losses_613731і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
≥2∞
__inference_loss_fn_0_613742П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥2∞
__inference_loss_fn_1_613753П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
њ2Љє
∞≤ђ
FullArgSpec/
args'Ъ$
jself
jx_in
je_in

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
њ2Љє
∞≤ђ
FullArgSpec/
args'Ъ$
jself
jx_in
je_in

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
њ2Љє
∞≤ђ
FullArgSpec/
args'Ъ$
jself
jx_in
je_in

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
њ2Љє
∞≤ђ
FullArgSpec/
args'Ъ$
jself
jx_in
je_in

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ѓ2Ђ
8__inference_batch_normalization_788_layer_call_fn_613784
8__inference_batch_normalization_788_layer_call_fn_613797і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
д2б
S__inference_batch_normalization_788_layer_call_and_return_conditional_losses_613815
S__inference_batch_normalization_788_layer_call_and_return_conditional_losses_613833і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ѓ2Ђ
8__inference_batch_normalization_789_layer_call_fn_613846
8__inference_batch_normalization_789_layer_call_fn_613859і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
д2б
S__inference_batch_normalization_789_layer_call_and_return_conditional_losses_613879
S__inference_batch_normalization_789_layer_call_and_return_conditional_losses_613913і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ѓ2Ђ
8__inference_batch_normalization_790_layer_call_fn_613926
8__inference_batch_normalization_790_layer_call_fn_613939і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
д2б
S__inference_batch_normalization_790_layer_call_and_return_conditional_losses_613959
S__inference_batch_normalization_790_layer_call_and_return_conditional_losses_613993і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ѓ2Ђ
8__inference_batch_normalization_791_layer_call_fn_614006
8__inference_batch_normalization_791_layer_call_fn_614019і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
д2б
S__inference_batch_normalization_791_layer_call_and_return_conditional_losses_614039
S__inference_batch_normalization_791_layer_call_and_return_conditional_losses_614073і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ѓ2Ђ
8__inference_batch_normalization_792_layer_call_fn_614086
8__inference_batch_normalization_792_layer_call_fn_614099і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
д2б
S__inference_batch_normalization_792_layer_call_and_return_conditional_losses_614117
S__inference_batch_normalization_792_layer_call_and_return_conditional_losses_614135і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ѓ2Ђ
8__inference_batch_normalization_793_layer_call_fn_614148
8__inference_batch_normalization_793_layer_call_fn_614161і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
д2б
S__inference_batch_normalization_793_layer_call_and_return_conditional_losses_614181
S__inference_batch_normalization_793_layer_call_and_return_conditional_losses_614215і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
≥2∞
__inference_loss_fn_2_614226П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ѓ2Ђ
8__inference_batch_normalization_794_layer_call_fn_614245
8__inference_batch_normalization_794_layer_call_fn_614258і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
д2б
S__inference_batch_normalization_794_layer_call_and_return_conditional_losses_614276
S__inference_batch_normalization_794_layer_call_and_return_conditional_losses_614294і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ѓ2Ђ
8__inference_batch_normalization_795_layer_call_fn_614307
8__inference_batch_normalization_795_layer_call_fn_614320і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
д2б
S__inference_batch_normalization_795_layer_call_and_return_conditional_losses_614340
S__inference_batch_normalization_795_layer_call_and_return_conditional_losses_614374і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ѓ2Ђ
8__inference_batch_normalization_796_layer_call_fn_614387
8__inference_batch_normalization_796_layer_call_fn_614400і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
д2б
S__inference_batch_normalization_796_layer_call_and_return_conditional_losses_614420
S__inference_batch_normalization_796_layer_call_and_return_conditional_losses_614454і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ѓ2Ђ
8__inference_batch_normalization_797_layer_call_fn_614467
8__inference_batch_normalization_797_layer_call_fn_614480і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
д2б
S__inference_batch_normalization_797_layer_call_and_return_conditional_losses_614500
S__inference_batch_normalization_797_layer_call_and_return_conditional_losses_614534і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ѓ2Ђ
8__inference_batch_normalization_798_layer_call_fn_614547
8__inference_batch_normalization_798_layer_call_fn_614560і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
д2б
S__inference_batch_normalization_798_layer_call_and_return_conditional_losses_614578
S__inference_batch_normalization_798_layer_call_and_return_conditional_losses_614596і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ѓ2Ђ
8__inference_batch_normalization_799_layer_call_fn_614609
8__inference_batch_normalization_799_layer_call_fn_614622і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
д2б
S__inference_batch_normalization_799_layer_call_and_return_conditional_losses_614642
S__inference_batch_normalization_799_layer_call_and_return_conditional_losses_614676і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
≥2∞
__inference_loss_fn_3_614687П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ ”
S__inference_batch_normalization_784_layer_call_and_return_conditional_losses_613549|)*('@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ ”
S__inference_batch_normalization_784_layer_call_and_return_conditional_losses_613583|)*('@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ Ђ
8__inference_batch_normalization_784_layer_call_fn_613516o)*('@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p 
™ "%К"€€€€€€€€€€€€€€€€€€Ђ
8__inference_batch_normalization_784_layer_call_fn_613529o)*('@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p
™ "%К"€€€€€€€€€€€€€€€€€€о
S__inference_batch_normalization_786_layer_call_and_return_conditional_losses_613633ЦGHIJMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€	
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€	
Ъ о
S__inference_batch_normalization_786_layer_call_and_return_conditional_losses_613651ЦGHIJMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€	
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€	
Ъ ∆
8__inference_batch_normalization_786_layer_call_fn_613602ЙGHIJMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€	
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€	∆
8__inference_batch_normalization_786_layer_call_fn_613615ЙGHIJMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€	
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€	”
S__inference_batch_normalization_787_layer_call_and_return_conditional_losses_613697|_`^]@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ ”
S__inference_batch_normalization_787_layer_call_and_return_conditional_losses_613731|_`^]@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ Ђ
8__inference_batch_normalization_787_layer_call_fn_613664o_`^]@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p 
™ "%К"€€€€€€€€€€€€€€€€€€Ђ
8__inference_batch_normalization_787_layer_call_fn_613677o_`^]@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p
™ "%К"€€€€€€€€€€€€€€€€€€р
S__inference_batch_normalization_788_layer_call_and_return_conditional_losses_613815ШxyЮЯMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ р
S__inference_batch_normalization_788_layer_call_and_return_conditional_losses_613833ШxyЮЯMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ »
8__inference_batch_normalization_788_layer_call_fn_613784ЛxyЮЯMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€»
8__inference_batch_normalization_788_layer_call_fn_613797ЛxyЮЯMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€’
S__inference_batch_normalization_789_layer_call_and_return_conditional_losses_613879~†°}|@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ ’
S__inference_batch_normalization_789_layer_call_and_return_conditional_losses_613913~†°}|@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ ≠
8__inference_batch_normalization_789_layer_call_fn_613846q†°}|@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p 
™ "%К"€€€€€€€€€€€€€€€€€€≠
8__inference_batch_normalization_789_layer_call_fn_613859q†°}|@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p
™ "%К"€€€€€€€€€€€€€€€€€€Ў
S__inference_batch_normalization_790_layer_call_and_return_conditional_losses_613959АҐ£БА@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ Ў
S__inference_batch_normalization_790_layer_call_and_return_conditional_losses_613993АҐ£БА@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ ѓ
8__inference_batch_normalization_790_layer_call_fn_613926sҐ£БА@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p 
™ "%К"€€€€€€€€€€€€€€€€€€ѓ
8__inference_batch_normalization_790_layer_call_fn_613939sҐ£БА@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p
™ "%К"€€€€€€€€€€€€€€€€€€Ў
S__inference_batch_normalization_791_layer_call_and_return_conditional_losses_614039А§•ЕД@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ Ў
S__inference_batch_normalization_791_layer_call_and_return_conditional_losses_614073А§•ЕД@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ ѓ
8__inference_batch_normalization_791_layer_call_fn_614006s§•ЕД@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p 
™ "%К"€€€€€€€€€€€€€€€€€€ѓ
8__inference_batch_normalization_791_layer_call_fn_614019s§•ЕД@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p
™ "%К"€€€€€€€€€€€€€€€€€€т
S__inference_batch_normalization_792_layer_call_and_return_conditional_losses_614117ЪЖЗ¶ІMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ т
S__inference_batch_normalization_792_layer_call_and_return_conditional_losses_614135ЪЖЗ¶ІMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ  
8__inference_batch_normalization_792_layer_call_fn_614086НЖЗ¶ІMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
8__inference_batch_normalization_792_layer_call_fn_614099НЖЗ¶ІMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€Ў
S__inference_batch_normalization_793_layer_call_and_return_conditional_losses_614181А®©ЙИ@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ Ў
S__inference_batch_normalization_793_layer_call_and_return_conditional_losses_614215А®©ЙИ@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ ѓ
8__inference_batch_normalization_793_layer_call_fn_614148s®©ЙИ@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p 
™ "%К"€€€€€€€€€€€€€€€€€€ѓ
8__inference_batch_normalization_793_layer_call_fn_614161s®©ЙИ@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p
™ "%К"€€€€€€€€€€€€€€€€€€т
S__inference_batch_normalization_794_layer_call_and_return_conditional_losses_614276ЪМН™ЂMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ т
S__inference_batch_normalization_794_layer_call_and_return_conditional_losses_614294ЪМН™ЂMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ  
8__inference_batch_normalization_794_layer_call_fn_614245НМН™ЂMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
8__inference_batch_normalization_794_layer_call_fn_614258НМН™ЂMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€Ў
S__inference_batch_normalization_795_layer_call_and_return_conditional_losses_614340Ађ≠СР@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ Ў
S__inference_batch_normalization_795_layer_call_and_return_conditional_losses_614374Ађ≠СР@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ ѓ
8__inference_batch_normalization_795_layer_call_fn_614307sђ≠СР@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p 
™ "%К"€€€€€€€€€€€€€€€€€€ѓ
8__inference_batch_normalization_795_layer_call_fn_614320sђ≠СР@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p
™ "%К"€€€€€€€€€€€€€€€€€€Ў
S__inference_batch_normalization_796_layer_call_and_return_conditional_losses_614420АЃѓХФ@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ Ў
S__inference_batch_normalization_796_layer_call_and_return_conditional_losses_614454АЃѓХФ@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ ѓ
8__inference_batch_normalization_796_layer_call_fn_614387sЃѓХФ@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p 
™ "%К"€€€€€€€€€€€€€€€€€€ѓ
8__inference_batch_normalization_796_layer_call_fn_614400sЃѓХФ@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p
™ "%К"€€€€€€€€€€€€€€€€€€Ў
S__inference_batch_normalization_797_layer_call_and_return_conditional_losses_614500А∞±ЩШ@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ Ў
S__inference_batch_normalization_797_layer_call_and_return_conditional_losses_614534А∞±ЩШ@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ ѓ
8__inference_batch_normalization_797_layer_call_fn_614467s∞±ЩШ@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p 
™ "%К"€€€€€€€€€€€€€€€€€€ѓ
8__inference_batch_normalization_797_layer_call_fn_614480s∞±ЩШ@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p
™ "%К"€€€€€€€€€€€€€€€€€€т
S__inference_batch_normalization_798_layer_call_and_return_conditional_losses_614578ЪЪЫ≤≥MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ т
S__inference_batch_normalization_798_layer_call_and_return_conditional_losses_614596ЪЪЫ≤≥MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ  
8__inference_batch_normalization_798_layer_call_fn_614547НЪЫ≤≥MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
8__inference_batch_normalization_798_layer_call_fn_614560НЪЫ≤≥MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€Ў
S__inference_batch_normalization_799_layer_call_and_return_conditional_losses_614642АіµЭЬ@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ Ў
S__inference_batch_normalization_799_layer_call_and_return_conditional_losses_614676АіµЭЬ@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€
Ъ ѓ
8__inference_batch_normalization_799_layer_call_fn_614609sіµЭЬ@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p 
™ "%К"€€€€€€€€€€€€€€€€€€ѓ
8__inference_batch_normalization_799_layer_call_fn_614622sіµЭЬ@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€
p
™ "%К"€€€€€€€€€€€€€€€€€€;
__inference_loss_fn_0_613742Ґ

Ґ 
™ "К ;
__inference_loss_fn_1_613753<Ґ

Ґ 
™ "К ;
__inference_loss_fn_2_614226vҐ

Ґ 
™ "К <
__inference_loss_fn_3_614687КҐ

Ґ 
™ "К н
$__inference_signature_wrapper_611223ƒК)*('7<=GHIJSvwxyЮЯz{†°}|~Ґ£БАВГ§•ЕДЖЗ¶І®©ЙИКЛМН™ЂОПђ≠СРТУЃѓХФЦЧ∞±ЩШЪЫ≤≥іµЭЬ_`^]lmфҐр
Ґ 
и™д
B
input_text13К0
input_text1€€€€€€€€€€€€€€€€€€ђ
N
input_text2?К<
input_text2+€€€€€€€€€€€€€€€€€€€€€€€€€€€
N
input_text3?К<
input_text3+€€€€€€€€€€€€€€€€€€€€€€€€€€€">™;
9
outputs.К+
outputs€€€€€€€€€€€€€€€€€€ї
__inference_tf_translate_611046ЧК)*('7<=GHIJSvwxyЮЯz{†°}|~Ґ£БАВГ§•ЕДЖЗ¶І®©ЙИКЛМН™ЂОПђ≠СРТУЃѓХФЦЧ∞±ЩШЪЫ≤≥іµЭЬ_`^]lm«Ґ√
їҐЈ
3К0
input_text1€€€€€€€€€€€€€€€€€€ђ
?К<
input_text2+€€€€€€€€€€€€€€€€€€€€€€€€€€€
?К<
input_text3+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ">™;
9
outputs.К+
outputs€€€€€€€€€€€€€€€€€€