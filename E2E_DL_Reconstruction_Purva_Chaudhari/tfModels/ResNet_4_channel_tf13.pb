
H
inputsPlaceholder*
dtype0*$
shape:���������}}
P
random_normal/shapeConst*%
valueB"            *
dtype0
?
random_normal/meanConst*
valueB
 *    *
dtype0
A
random_normal/stddevConst*
valueB
 *  �?*
dtype0
~
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
T0*
dtype0*
seed2 *

seed 
[
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
T0
D
random_normalAddrandom_normal/mulrandom_normal/mean*
T0
�
layer1Conv2Dinputsrandom_normal*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
C
random_normal_1/shapeConst*
valueB:*
dtype0
A
random_normal_1/meanConst*
valueB
 *    *
dtype0
C
random_normal_1/stddevConst*
valueB
 *  �?*
dtype0
�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape*
dtype0*
seed2 *

seed *
T0
a
random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*
T0
J
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*
T0
K
BiasAddBiasAddlayer1random_normal_1*
data_formatNHWC*
T0

conv1ReluBiasAdd*
T0
u
maxpool1MaxPoolconv1*
ksize
*
paddingSAME*
T0*
data_formatNHWC*
strides

R
random_normal_2/shapeConst*%
valueB"            *
dtype0
A
random_normal_2/meanConst*
valueB
 *    *
dtype0
C
random_normal_2/stddevConst*
valueB
 *  �?*
dtype0
�
$random_normal_2/RandomStandardNormalRandomStandardNormalrandom_normal_2/shape*
T0*
dtype0*
seed2 *

seed 
a
random_normal_2/mulMul$random_normal_2/RandomStandardNormalrandom_normal_2/stddev*
T0
J
random_normal_2Addrandom_normal_2/mulrandom_normal_2/mean*
T0
�
Conv2DConv2Dmaxpool1random_normal_2*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
C
random_normal_3/shapeConst*
valueB:*
dtype0
A
random_normal_3/meanConst*
valueB
 *    *
dtype0
C
random_normal_3/stddevConst*
valueB
 *  �?*
dtype0
�
$random_normal_3/RandomStandardNormalRandomStandardNormalrandom_normal_3/shape*

seed *
T0*
dtype0*
seed2 
a
random_normal_3/mulMul$random_normal_3/RandomStandardNormalrandom_normal_3/stddev*
T0
J
random_normal_3Addrandom_normal_3/mulrandom_normal_3/mean*
T0
M
	BiasAdd_1BiasAddConv2Drandom_normal_3*
T0*
data_formatNHWC
'
blockLayer1Relu	BiasAdd_1*
T0
R
random_normal_4/shapeConst*%
valueB"            *
dtype0
A
random_normal_4/meanConst*
valueB
 *    *
dtype0
C
random_normal_4/stddevConst*
valueB
 *  �?*
dtype0
�
$random_normal_4/RandomStandardNormalRandomStandardNormalrandom_normal_4/shape*

seed *
T0*
dtype0*
seed2 
a
random_normal_4/mulMul$random_normal_4/RandomStandardNormalrandom_normal_4/stddev*
T0
J
random_normal_4Addrandom_normal_4/mulrandom_normal_4/mean*
T0
�
blockLayer2Conv2DblockLayer1random_normal_4*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
C
random_normal_5/shapeConst*
valueB:*
dtype0
A
random_normal_5/meanConst*
valueB
 *    *
dtype0
C
random_normal_5/stddevConst*
valueB
 *  �?*
dtype0
�
$random_normal_5/RandomStandardNormalRandomStandardNormalrandom_normal_5/shape*

seed *
T0*
dtype0*
seed2 
a
random_normal_5/mulMul$random_normal_5/RandomStandardNormalrandom_normal_5/stddev*
T0
J
random_normal_5Addrandom_normal_5/mulrandom_normal_5/mean*
T0
R
	BiasAdd_2BiasAddblockLayer2random_normal_5*
T0*
data_formatNHWC
0
Cast/xConst*
value	B
 Z *
dtype0

5
blockLayer3/SwitchSwitchCast/xCast/x*
T0

?
blockLayer3/switch_tIdentityblockLayer3/Switch:1*
T0

=
blockLayer3/switch_fIdentityblockLayer3/Switch*
T0

0
blockLayer3/pred_idIdentityCast/x*
T0

s
blockLayer3/random_normal/shapeConst^blockLayer3/switch_t*
dtype0*%
valueB"            
b
blockLayer3/random_normal/meanConst^blockLayer3/switch_t*
valueB
 *    *
dtype0
d
 blockLayer3/random_normal/stddevConst^blockLayer3/switch_t*
valueB
 *  �?*
dtype0
�
.blockLayer3/random_normal/RandomStandardNormalRandomStandardNormalblockLayer3/random_normal/shape*
T0*
dtype0*
seed2 *

seed 

blockLayer3/random_normal/mulMul.blockLayer3/random_normal/RandomStandardNormal blockLayer3/random_normal/stddev*
T0
h
blockLayer3/random_normalAddblockLayer3/random_normal/mulblockLayer3/random_normal/mean*
T0
�
blockLayer3/Conv2DConv2DblockLayer3/Conv2D/Switch:1blockLayer3/random_normal*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
h
blockLayer3/Conv2D/SwitchSwitchmaxpool1blockLayer3/pred_id*
T0*
_class
loc:@maxpool1
f
!blockLayer3/random_normal_1/shapeConst^blockLayer3/switch_t*
valueB:*
dtype0
d
 blockLayer3/random_normal_1/meanConst^blockLayer3/switch_t*
valueB
 *    *
dtype0
f
"blockLayer3/random_normal_1/stddevConst^blockLayer3/switch_t*
valueB
 *  �?*
dtype0
�
0blockLayer3/random_normal_1/RandomStandardNormalRandomStandardNormal!blockLayer3/random_normal_1/shape*

seed *
T0*
dtype0*
seed2 
�
blockLayer3/random_normal_1/mulMul0blockLayer3/random_normal_1/RandomStandardNormal"blockLayer3/random_normal_1/stddev*
T0
n
blockLayer3/random_normal_1AddblockLayer3/random_normal_1/mul blockLayer3/random_normal_1/mean*
T0
o
blockLayer3/BiasAddBiasAddblockLayer3/Conv2DblockLayer3/random_normal_1*
T0*
data_formatNHWC
N
blockLayer3/AddAddblockLayer3/Add/Switch:1blockLayer3/BiasAdd*
T0
g
blockLayer3/Add/SwitchSwitch	BiasAdd_2blockLayer3/pred_id*
T0*
_class
loc:@BiasAdd_2
e
blockLayer3/Switch_1Switch	BiasAdd_2blockLayer3/pred_id*
T0*
_class
loc:@BiasAdd_2
S
blockLayer3/MergeMergeblockLayer3/Switch_1blockLayer3/Add*
T0*
N
4
blockOutputLayerRelublockLayer3/Merge*
T0
R
random_normal_6/shapeConst*%
valueB"            *
dtype0
A
random_normal_6/meanConst*
valueB
 *    *
dtype0
C
random_normal_6/stddevConst*
valueB
 *  �?*
dtype0
�
$random_normal_6/RandomStandardNormalRandomStandardNormalrandom_normal_6/shape*
dtype0*
seed2 *

seed *
T0
a
random_normal_6/mulMul$random_normal_6/RandomStandardNormalrandom_normal_6/stddev*
T0
J
random_normal_6Addrandom_normal_6/mulrandom_normal_6/mean*
T0
�
Conv2D_1Conv2DblockOutputLayerrandom_normal_6*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
C
random_normal_7/shapeConst*
valueB:*
dtype0
A
random_normal_7/meanConst*
valueB
 *    *
dtype0
C
random_normal_7/stddevConst*
valueB
 *  �?*
dtype0
�
$random_normal_7/RandomStandardNormalRandomStandardNormalrandom_normal_7/shape*
T0*
dtype0*
seed2 *

seed 
a
random_normal_7/mulMul$random_normal_7/RandomStandardNormalrandom_normal_7/stddev*
T0
J
random_normal_7Addrandom_normal_7/mulrandom_normal_7/mean*
T0
O
	BiasAdd_3BiasAddConv2D_1random_normal_7*
data_formatNHWC*
T0
)
blockLayer1_1Relu	BiasAdd_3*
T0
R
random_normal_8/shapeConst*%
valueB"            *
dtype0
A
random_normal_8/meanConst*
valueB
 *    *
dtype0
C
random_normal_8/stddevConst*
valueB
 *  �?*
dtype0
�
$random_normal_8/RandomStandardNormalRandomStandardNormalrandom_normal_8/shape*

seed *
T0*
dtype0*
seed2 
a
random_normal_8/mulMul$random_normal_8/RandomStandardNormalrandom_normal_8/stddev*
T0
J
random_normal_8Addrandom_normal_8/mulrandom_normal_8/mean*
T0
�
blockLayer2_1Conv2DblockLayer1_1random_normal_8*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0
C
random_normal_9/shapeConst*
valueB:*
dtype0
A
random_normal_9/meanConst*
valueB
 *    *
dtype0
C
random_normal_9/stddevConst*
valueB
 *  �?*
dtype0
�
$random_normal_9/RandomStandardNormalRandomStandardNormalrandom_normal_9/shape*
T0*
dtype0*
seed2 *

seed 
a
random_normal_9/mulMul$random_normal_9/RandomStandardNormalrandom_normal_9/stddev*
T0
J
random_normal_9Addrandom_normal_9/mulrandom_normal_9/mean*
T0
T
	BiasAdd_4BiasAddblockLayer2_1random_normal_9*
T0*
data_formatNHWC
2
Cast_1/xConst*
value	B
 Z *
dtype0

;
blockLayer3_1/SwitchSwitchCast_1/xCast_1/x*
T0

C
blockLayer3_1/switch_tIdentityblockLayer3_1/Switch:1*
T0

A
blockLayer3_1/switch_fIdentityblockLayer3_1/Switch*
T0

4
blockLayer3_1/pred_idIdentityCast_1/x*
T0

w
!blockLayer3_1/random_normal/shapeConst^blockLayer3_1/switch_t*%
valueB"            *
dtype0
f
 blockLayer3_1/random_normal/meanConst^blockLayer3_1/switch_t*
dtype0*
valueB
 *    
h
"blockLayer3_1/random_normal/stddevConst^blockLayer3_1/switch_t*
valueB
 *  �?*
dtype0
�
0blockLayer3_1/random_normal/RandomStandardNormalRandomStandardNormal!blockLayer3_1/random_normal/shape*
T0*
dtype0*
seed2 *

seed 
�
blockLayer3_1/random_normal/mulMul0blockLayer3_1/random_normal/RandomStandardNormal"blockLayer3_1/random_normal/stddev*
T0
n
blockLayer3_1/random_normalAddblockLayer3_1/random_normal/mul blockLayer3_1/random_normal/mean*
T0
�
blockLayer3_1/Conv2DConv2DblockLayer3_1/Conv2D/Switch:1blockLayer3_1/random_normal*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0
|
blockLayer3_1/Conv2D/SwitchSwitchblockOutputLayerblockLayer3_1/pred_id*
T0*#
_class
loc:@blockOutputLayer
j
#blockLayer3_1/random_normal_1/shapeConst^blockLayer3_1/switch_t*
valueB:*
dtype0
h
"blockLayer3_1/random_normal_1/meanConst^blockLayer3_1/switch_t*
dtype0*
valueB
 *    
j
$blockLayer3_1/random_normal_1/stddevConst^blockLayer3_1/switch_t*
valueB
 *  �?*
dtype0
�
2blockLayer3_1/random_normal_1/RandomStandardNormalRandomStandardNormal#blockLayer3_1/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0
�
!blockLayer3_1/random_normal_1/mulMul2blockLayer3_1/random_normal_1/RandomStandardNormal$blockLayer3_1/random_normal_1/stddev*
T0
t
blockLayer3_1/random_normal_1Add!blockLayer3_1/random_normal_1/mul"blockLayer3_1/random_normal_1/mean*
T0
u
blockLayer3_1/BiasAddBiasAddblockLayer3_1/Conv2DblockLayer3_1/random_normal_1*
T0*
data_formatNHWC
T
blockLayer3_1/AddAddblockLayer3_1/Add/Switch:1blockLayer3_1/BiasAdd*
T0
k
blockLayer3_1/Add/SwitchSwitch	BiasAdd_4blockLayer3_1/pred_id*
T0*
_class
loc:@BiasAdd_4
i
blockLayer3_1/Switch_1Switch	BiasAdd_4blockLayer3_1/pred_id*
T0*
_class
loc:@BiasAdd_4
Y
blockLayer3_1/MergeMergeblockLayer3_1/Switch_1blockLayer3_1/Add*
T0*
N
8
blockOutputLayer_1RelublockLayer3_1/Merge*
T0
S
random_normal_10/shapeConst*%
valueB"            *
dtype0
B
random_normal_10/meanConst*
valueB
 *    *
dtype0
D
random_normal_10/stddevConst*
valueB
 *  �?*
dtype0
�
%random_normal_10/RandomStandardNormalRandomStandardNormalrandom_normal_10/shape*
dtype0*
seed2 *

seed *
T0
d
random_normal_10/mulMul%random_normal_10/RandomStandardNormalrandom_normal_10/stddev*
T0
M
random_normal_10Addrandom_normal_10/mulrandom_normal_10/mean*
T0
�
Conv2D_2Conv2DblockOutputLayer_1random_normal_10*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
D
random_normal_11/shapeConst*
valueB:*
dtype0
B
random_normal_11/meanConst*
valueB
 *    *
dtype0
D
random_normal_11/stddevConst*
valueB
 *  �?*
dtype0
�
%random_normal_11/RandomStandardNormalRandomStandardNormalrandom_normal_11/shape*
T0*
dtype0*
seed2 *

seed 
d
random_normal_11/mulMul%random_normal_11/RandomStandardNormalrandom_normal_11/stddev*
T0
M
random_normal_11Addrandom_normal_11/mulrandom_normal_11/mean*
T0
P
	BiasAdd_5BiasAddConv2D_2random_normal_11*
T0*
data_formatNHWC
)
blockLayer1_2Relu	BiasAdd_5*
T0
S
random_normal_12/shapeConst*%
valueB"            *
dtype0
B
random_normal_12/meanConst*
valueB
 *    *
dtype0
D
random_normal_12/stddevConst*
valueB
 *  �?*
dtype0
�
%random_normal_12/RandomStandardNormalRandomStandardNormalrandom_normal_12/shape*
dtype0*
seed2 *

seed *
T0
d
random_normal_12/mulMul%random_normal_12/RandomStandardNormalrandom_normal_12/stddev*
T0
M
random_normal_12Addrandom_normal_12/mulrandom_normal_12/mean*
T0
�
blockLayer2_2Conv2DblockLayer1_2random_normal_12*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
D
random_normal_13/shapeConst*
dtype0*
valueB:
B
random_normal_13/meanConst*
valueB
 *    *
dtype0
D
random_normal_13/stddevConst*
valueB
 *  �?*
dtype0
�
%random_normal_13/RandomStandardNormalRandomStandardNormalrandom_normal_13/shape*
dtype0*
seed2 *

seed *
T0
d
random_normal_13/mulMul%random_normal_13/RandomStandardNormalrandom_normal_13/stddev*
T0
M
random_normal_13Addrandom_normal_13/mulrandom_normal_13/mean*
T0
U
	BiasAdd_6BiasAddblockLayer2_2random_normal_13*
T0*
data_formatNHWC
2
Cast_2/xConst*
value	B
 Z *
dtype0

;
blockLayer3_2/SwitchSwitchCast_2/xCast_2/x*
T0

C
blockLayer3_2/switch_tIdentityblockLayer3_2/Switch:1*
T0

A
blockLayer3_2/switch_fIdentityblockLayer3_2/Switch*
T0

4
blockLayer3_2/pred_idIdentityCast_2/x*
T0

w
!blockLayer3_2/random_normal/shapeConst^blockLayer3_2/switch_t*%
valueB"            *
dtype0
f
 blockLayer3_2/random_normal/meanConst^blockLayer3_2/switch_t*
dtype0*
valueB
 *    
h
"blockLayer3_2/random_normal/stddevConst^blockLayer3_2/switch_t*
valueB
 *  �?*
dtype0
�
0blockLayer3_2/random_normal/RandomStandardNormalRandomStandardNormal!blockLayer3_2/random_normal/shape*
T0*
dtype0*
seed2 *

seed 
�
blockLayer3_2/random_normal/mulMul0blockLayer3_2/random_normal/RandomStandardNormal"blockLayer3_2/random_normal/stddev*
T0
n
blockLayer3_2/random_normalAddblockLayer3_2/random_normal/mul blockLayer3_2/random_normal/mean*
T0
�
blockLayer3_2/Conv2DConv2DblockLayer3_2/Conv2D/Switch:1blockLayer3_2/random_normal*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
blockLayer3_2/Conv2D/SwitchSwitchblockOutputLayer_1blockLayer3_2/pred_id*
T0*%
_class
loc:@blockOutputLayer_1
j
#blockLayer3_2/random_normal_1/shapeConst^blockLayer3_2/switch_t*
valueB:*
dtype0
h
"blockLayer3_2/random_normal_1/meanConst^blockLayer3_2/switch_t*
valueB
 *    *
dtype0
j
$blockLayer3_2/random_normal_1/stddevConst^blockLayer3_2/switch_t*
valueB
 *  �?*
dtype0
�
2blockLayer3_2/random_normal_1/RandomStandardNormalRandomStandardNormal#blockLayer3_2/random_normal_1/shape*

seed *
T0*
dtype0*
seed2 
�
!blockLayer3_2/random_normal_1/mulMul2blockLayer3_2/random_normal_1/RandomStandardNormal$blockLayer3_2/random_normal_1/stddev*
T0
t
blockLayer3_2/random_normal_1Add!blockLayer3_2/random_normal_1/mul"blockLayer3_2/random_normal_1/mean*
T0
u
blockLayer3_2/BiasAddBiasAddblockLayer3_2/Conv2DblockLayer3_2/random_normal_1*
T0*
data_formatNHWC
T
blockLayer3_2/AddAddblockLayer3_2/Add/Switch:1blockLayer3_2/BiasAdd*
T0
k
blockLayer3_2/Add/SwitchSwitch	BiasAdd_6blockLayer3_2/pred_id*
_class
loc:@BiasAdd_6*
T0
i
blockLayer3_2/Switch_1Switch	BiasAdd_6blockLayer3_2/pred_id*
T0*
_class
loc:@BiasAdd_6
Y
blockLayer3_2/MergeMergeblockLayer3_2/Switch_1blockLayer3_2/Add*
N*
T0
8
blockOutputLayer_2RelublockLayer3_2/Merge*
T0
S
random_normal_14/shapeConst*%
valueB"             *
dtype0
B
random_normal_14/meanConst*
dtype0*
valueB
 *    
D
random_normal_14/stddevConst*
valueB
 *  �?*
dtype0
�
%random_normal_14/RandomStandardNormalRandomStandardNormalrandom_normal_14/shape*
T0*
dtype0*
seed2 *

seed 
d
random_normal_14/mulMul%random_normal_14/RandomStandardNormalrandom_normal_14/stddev*
T0
M
random_normal_14Addrandom_normal_14/mulrandom_normal_14/mean*
T0
�
Conv2D_3Conv2DblockOutputLayer_2random_normal_14*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
D
random_normal_15/shapeConst*
valueB: *
dtype0
B
random_normal_15/meanConst*
dtype0*
valueB
 *    
D
random_normal_15/stddevConst*
valueB
 *  �?*
dtype0
�
%random_normal_15/RandomStandardNormalRandomStandardNormalrandom_normal_15/shape*

seed *
T0*
dtype0*
seed2 
d
random_normal_15/mulMul%random_normal_15/RandomStandardNormalrandom_normal_15/stddev*
T0
M
random_normal_15Addrandom_normal_15/mulrandom_normal_15/mean*
T0
P
	BiasAdd_7BiasAddConv2D_3random_normal_15*
T0*
data_formatNHWC
)
blockLayer1_3Relu	BiasAdd_7*
T0
S
random_normal_16/shapeConst*%
valueB"              *
dtype0
B
random_normal_16/meanConst*
valueB
 *    *
dtype0
D
random_normal_16/stddevConst*
valueB
 *  �?*
dtype0
�
%random_normal_16/RandomStandardNormalRandomStandardNormalrandom_normal_16/shape*
dtype0*
seed2 *

seed *
T0
d
random_normal_16/mulMul%random_normal_16/RandomStandardNormalrandom_normal_16/stddev*
T0
M
random_normal_16Addrandom_normal_16/mulrandom_normal_16/mean*
T0
�
blockLayer2_3Conv2DblockLayer1_3random_normal_16*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
D
random_normal_17/shapeConst*
valueB: *
dtype0
B
random_normal_17/meanConst*
valueB
 *    *
dtype0
D
random_normal_17/stddevConst*
dtype0*
valueB
 *  �?
�
%random_normal_17/RandomStandardNormalRandomStandardNormalrandom_normal_17/shape*
T0*
dtype0*
seed2 *

seed 
d
random_normal_17/mulMul%random_normal_17/RandomStandardNormalrandom_normal_17/stddev*
T0
M
random_normal_17Addrandom_normal_17/mulrandom_normal_17/mean*
T0
U
	BiasAdd_8BiasAddblockLayer2_3random_normal_17*
T0*
data_formatNHWC
2
Cast_3/xConst*
value	B
 Z*
dtype0

;
blockLayer3_3/SwitchSwitchCast_3/xCast_3/x*
T0

C
blockLayer3_3/switch_tIdentityblockLayer3_3/Switch:1*
T0

A
blockLayer3_3/switch_fIdentityblockLayer3_3/Switch*
T0

4
blockLayer3_3/pred_idIdentityCast_3/x*
T0

w
!blockLayer3_3/random_normal/shapeConst^blockLayer3_3/switch_t*%
valueB"             *
dtype0
f
 blockLayer3_3/random_normal/meanConst^blockLayer3_3/switch_t*
valueB
 *    *
dtype0
h
"blockLayer3_3/random_normal/stddevConst^blockLayer3_3/switch_t*
valueB
 *  �?*
dtype0
�
0blockLayer3_3/random_normal/RandomStandardNormalRandomStandardNormal!blockLayer3_3/random_normal/shape*
seed2 *

seed *
T0*
dtype0
�
blockLayer3_3/random_normal/mulMul0blockLayer3_3/random_normal/RandomStandardNormal"blockLayer3_3/random_normal/stddev*
T0
n
blockLayer3_3/random_normalAddblockLayer3_3/random_normal/mul blockLayer3_3/random_normal/mean*
T0
�
blockLayer3_3/Conv2DConv2DblockLayer3_3/Conv2D/Switch:1blockLayer3_3/random_normal*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
blockLayer3_3/Conv2D/SwitchSwitchblockOutputLayer_2blockLayer3_3/pred_id*
T0*%
_class
loc:@blockOutputLayer_2
j
#blockLayer3_3/random_normal_1/shapeConst^blockLayer3_3/switch_t*
valueB: *
dtype0
h
"blockLayer3_3/random_normal_1/meanConst^blockLayer3_3/switch_t*
valueB
 *    *
dtype0
j
$blockLayer3_3/random_normal_1/stddevConst^blockLayer3_3/switch_t*
valueB
 *  �?*
dtype0
�
2blockLayer3_3/random_normal_1/RandomStandardNormalRandomStandardNormal#blockLayer3_3/random_normal_1/shape*

seed *
T0*
dtype0*
seed2 
�
!blockLayer3_3/random_normal_1/mulMul2blockLayer3_3/random_normal_1/RandomStandardNormal$blockLayer3_3/random_normal_1/stddev*
T0
t
blockLayer3_3/random_normal_1Add!blockLayer3_3/random_normal_1/mul"blockLayer3_3/random_normal_1/mean*
T0
u
blockLayer3_3/BiasAddBiasAddblockLayer3_3/Conv2DblockLayer3_3/random_normal_1*
data_formatNHWC*
T0
T
blockLayer3_3/AddAddblockLayer3_3/Add/Switch:1blockLayer3_3/BiasAdd*
T0
k
blockLayer3_3/Add/SwitchSwitch	BiasAdd_8blockLayer3_3/pred_id*
T0*
_class
loc:@BiasAdd_8
i
blockLayer3_3/Switch_1Switch	BiasAdd_8blockLayer3_3/pred_id*
_class
loc:@BiasAdd_8*
T0
Y
blockLayer3_3/MergeMergeblockLayer3_3/Switch_1blockLayer3_3/Add*
N*
T0
8
blockOutputLayer_3RelublockLayer3_3/Merge*
T0
S
random_normal_18/shapeConst*%
valueB"              *
dtype0
B
random_normal_18/meanConst*
valueB
 *    *
dtype0
D
random_normal_18/stddevConst*
valueB
 *  �?*
dtype0
�
%random_normal_18/RandomStandardNormalRandomStandardNormalrandom_normal_18/shape*
T0*
dtype0*
seed2 *

seed 
d
random_normal_18/mulMul%random_normal_18/RandomStandardNormalrandom_normal_18/stddev*
T0
M
random_normal_18Addrandom_normal_18/mulrandom_normal_18/mean*
T0
�
Conv2D_4Conv2DblockOutputLayer_3random_normal_18*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
D
random_normal_19/shapeConst*
valueB: *
dtype0
B
random_normal_19/meanConst*
valueB
 *    *
dtype0
D
random_normal_19/stddevConst*
valueB
 *  �?*
dtype0
�
%random_normal_19/RandomStandardNormalRandomStandardNormalrandom_normal_19/shape*
dtype0*
seed2 *

seed *
T0
d
random_normal_19/mulMul%random_normal_19/RandomStandardNormalrandom_normal_19/stddev*
T0
M
random_normal_19Addrandom_normal_19/mulrandom_normal_19/mean*
T0
P
	BiasAdd_9BiasAddConv2D_4random_normal_19*
T0*
data_formatNHWC
)
blockLayer1_4Relu	BiasAdd_9*
T0
S
random_normal_20/shapeConst*%
valueB"              *
dtype0
B
random_normal_20/meanConst*
valueB
 *    *
dtype0
D
random_normal_20/stddevConst*
valueB
 *  �?*
dtype0
�
%random_normal_20/RandomStandardNormalRandomStandardNormalrandom_normal_20/shape*
T0*
dtype0*
seed2 *

seed 
d
random_normal_20/mulMul%random_normal_20/RandomStandardNormalrandom_normal_20/stddev*
T0
M
random_normal_20Addrandom_normal_20/mulrandom_normal_20/mean*
T0
�
blockLayer2_4Conv2DblockLayer1_4random_normal_20*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
D
random_normal_21/shapeConst*
valueB: *
dtype0
B
random_normal_21/meanConst*
valueB
 *    *
dtype0
D
random_normal_21/stddevConst*
valueB
 *  �?*
dtype0
�
%random_normal_21/RandomStandardNormalRandomStandardNormalrandom_normal_21/shape*
T0*
dtype0*
seed2 *

seed 
d
random_normal_21/mulMul%random_normal_21/RandomStandardNormalrandom_normal_21/stddev*
T0
M
random_normal_21Addrandom_normal_21/mulrandom_normal_21/mean*
T0
V

BiasAdd_10BiasAddblockLayer2_4random_normal_21*
T0*
data_formatNHWC
2
Cast_4/xConst*
value	B
 Z *
dtype0

;
blockLayer3_4/SwitchSwitchCast_4/xCast_4/x*
T0

C
blockLayer3_4/switch_tIdentityblockLayer3_4/Switch:1*
T0

A
blockLayer3_4/switch_fIdentityblockLayer3_4/Switch*
T0

4
blockLayer3_4/pred_idIdentityCast_4/x*
T0

w
!blockLayer3_4/random_normal/shapeConst^blockLayer3_4/switch_t*
dtype0*%
valueB"              
f
 blockLayer3_4/random_normal/meanConst^blockLayer3_4/switch_t*
valueB
 *    *
dtype0
h
"blockLayer3_4/random_normal/stddevConst^blockLayer3_4/switch_t*
valueB
 *  �?*
dtype0
�
0blockLayer3_4/random_normal/RandomStandardNormalRandomStandardNormal!blockLayer3_4/random_normal/shape*

seed *
T0*
dtype0*
seed2 
�
blockLayer3_4/random_normal/mulMul0blockLayer3_4/random_normal/RandomStandardNormal"blockLayer3_4/random_normal/stddev*
T0
n
blockLayer3_4/random_normalAddblockLayer3_4/random_normal/mul blockLayer3_4/random_normal/mean*
T0
�
blockLayer3_4/Conv2DConv2DblockLayer3_4/Conv2D/Switch:1blockLayer3_4/random_normal*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0
�
blockLayer3_4/Conv2D/SwitchSwitchblockOutputLayer_3blockLayer3_4/pred_id*
T0*%
_class
loc:@blockOutputLayer_3
j
#blockLayer3_4/random_normal_1/shapeConst^blockLayer3_4/switch_t*
valueB: *
dtype0
h
"blockLayer3_4/random_normal_1/meanConst^blockLayer3_4/switch_t*
valueB
 *    *
dtype0
j
$blockLayer3_4/random_normal_1/stddevConst^blockLayer3_4/switch_t*
valueB
 *  �?*
dtype0
�
2blockLayer3_4/random_normal_1/RandomStandardNormalRandomStandardNormal#blockLayer3_4/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0
�
!blockLayer3_4/random_normal_1/mulMul2blockLayer3_4/random_normal_1/RandomStandardNormal$blockLayer3_4/random_normal_1/stddev*
T0
t
blockLayer3_4/random_normal_1Add!blockLayer3_4/random_normal_1/mul"blockLayer3_4/random_normal_1/mean*
T0
u
blockLayer3_4/BiasAddBiasAddblockLayer3_4/Conv2DblockLayer3_4/random_normal_1*
T0*
data_formatNHWC
T
blockLayer3_4/AddAddblockLayer3_4/Add/Switch:1blockLayer3_4/BiasAdd*
T0
m
blockLayer3_4/Add/SwitchSwitch
BiasAdd_10blockLayer3_4/pred_id*
T0*
_class
loc:@BiasAdd_10
k
blockLayer3_4/Switch_1Switch
BiasAdd_10blockLayer3_4/pred_id*
T0*
_class
loc:@BiasAdd_10
Y
blockLayer3_4/MergeMergeblockLayer3_4/Switch_1blockLayer3_4/Add*
T0*
N
8
blockOutputLayer_4RelublockLayer3_4/Merge*
T0
S
random_normal_22/shapeConst*%
valueB"              *
dtype0
B
random_normal_22/meanConst*
valueB
 *    *
dtype0
D
random_normal_22/stddevConst*
valueB
 *  �?*
dtype0
�
%random_normal_22/RandomStandardNormalRandomStandardNormalrandom_normal_22/shape*
seed2 *

seed *
T0*
dtype0
d
random_normal_22/mulMul%random_normal_22/RandomStandardNormalrandom_normal_22/stddev*
T0
M
random_normal_22Addrandom_normal_22/mulrandom_normal_22/mean*
T0
�
Conv2D_5Conv2DblockOutputLayer_4random_normal_22*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0
D
random_normal_23/shapeConst*
valueB: *
dtype0
B
random_normal_23/meanConst*
valueB
 *    *
dtype0
D
random_normal_23/stddevConst*
valueB
 *  �?*
dtype0
�
%random_normal_23/RandomStandardNormalRandomStandardNormalrandom_normal_23/shape*
T0*
dtype0*
seed2 *

seed 
d
random_normal_23/mulMul%random_normal_23/RandomStandardNormalrandom_normal_23/stddev*
T0
M
random_normal_23Addrandom_normal_23/mulrandom_normal_23/mean*
T0
Q

BiasAdd_11BiasAddConv2D_5random_normal_23*
data_formatNHWC*
T0
*
blockLayer1_5Relu
BiasAdd_11*
T0
S
random_normal_24/shapeConst*%
valueB"              *
dtype0
B
random_normal_24/meanConst*
valueB
 *    *
dtype0
D
random_normal_24/stddevConst*
valueB
 *  �?*
dtype0
�
%random_normal_24/RandomStandardNormalRandomStandardNormalrandom_normal_24/shape*
dtype0*
seed2 *

seed *
T0
d
random_normal_24/mulMul%random_normal_24/RandomStandardNormalrandom_normal_24/stddev*
T0
M
random_normal_24Addrandom_normal_24/mulrandom_normal_24/mean*
T0
�
blockLayer2_5Conv2DblockLayer1_5random_normal_24*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
D
random_normal_25/shapeConst*
valueB: *
dtype0
B
random_normal_25/meanConst*
valueB
 *    *
dtype0
D
random_normal_25/stddevConst*
valueB
 *  �?*
dtype0
�
%random_normal_25/RandomStandardNormalRandomStandardNormalrandom_normal_25/shape*
dtype0*
seed2 *

seed *
T0
d
random_normal_25/mulMul%random_normal_25/RandomStandardNormalrandom_normal_25/stddev*
T0
M
random_normal_25Addrandom_normal_25/mulrandom_normal_25/mean*
T0
V

BiasAdd_12BiasAddblockLayer2_5random_normal_25*
T0*
data_formatNHWC
2
Cast_5/xConst*
value	B
 Z *
dtype0

;
blockLayer3_5/SwitchSwitchCast_5/xCast_5/x*
T0

C
blockLayer3_5/switch_tIdentityblockLayer3_5/Switch:1*
T0

A
blockLayer3_5/switch_fIdentityblockLayer3_5/Switch*
T0

4
blockLayer3_5/pred_idIdentityCast_5/x*
T0

w
!blockLayer3_5/random_normal/shapeConst^blockLayer3_5/switch_t*%
valueB"              *
dtype0
f
 blockLayer3_5/random_normal/meanConst^blockLayer3_5/switch_t*
valueB
 *    *
dtype0
h
"blockLayer3_5/random_normal/stddevConst^blockLayer3_5/switch_t*
valueB
 *  �?*
dtype0
�
0blockLayer3_5/random_normal/RandomStandardNormalRandomStandardNormal!blockLayer3_5/random_normal/shape*
T0*
dtype0*
seed2 *

seed 
�
blockLayer3_5/random_normal/mulMul0blockLayer3_5/random_normal/RandomStandardNormal"blockLayer3_5/random_normal/stddev*
T0
n
blockLayer3_5/random_normalAddblockLayer3_5/random_normal/mul blockLayer3_5/random_normal/mean*
T0
�
blockLayer3_5/Conv2DConv2DblockLayer3_5/Conv2D/Switch:1blockLayer3_5/random_normal*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides

�
blockLayer3_5/Conv2D/SwitchSwitchblockOutputLayer_4blockLayer3_5/pred_id*
T0*%
_class
loc:@blockOutputLayer_4
j
#blockLayer3_5/random_normal_1/shapeConst^blockLayer3_5/switch_t*
valueB: *
dtype0
h
"blockLayer3_5/random_normal_1/meanConst^blockLayer3_5/switch_t*
valueB
 *    *
dtype0
j
$blockLayer3_5/random_normal_1/stddevConst^blockLayer3_5/switch_t*
valueB
 *  �?*
dtype0
�
2blockLayer3_5/random_normal_1/RandomStandardNormalRandomStandardNormal#blockLayer3_5/random_normal_1/shape*
seed2 *

seed *
T0*
dtype0
�
!blockLayer3_5/random_normal_1/mulMul2blockLayer3_5/random_normal_1/RandomStandardNormal$blockLayer3_5/random_normal_1/stddev*
T0
t
blockLayer3_5/random_normal_1Add!blockLayer3_5/random_normal_1/mul"blockLayer3_5/random_normal_1/mean*
T0
u
blockLayer3_5/BiasAddBiasAddblockLayer3_5/Conv2DblockLayer3_5/random_normal_1*
T0*
data_formatNHWC
T
blockLayer3_5/AddAddblockLayer3_5/Add/Switch:1blockLayer3_5/BiasAdd*
T0
m
blockLayer3_5/Add/SwitchSwitch
BiasAdd_12blockLayer3_5/pred_id*
T0*
_class
loc:@BiasAdd_12
k
blockLayer3_5/Switch_1Switch
BiasAdd_12blockLayer3_5/pred_id*
_class
loc:@BiasAdd_12*
T0
Y
blockLayer3_5/MergeMergeblockLayer3_5/Switch_1blockLayer3_5/Add*
T0*
N
8
blockOutputLayer_5RelublockLayer3_5/Merge*
T0
S
random_normal_26/shapeConst*%
valueB"              *
dtype0
B
random_normal_26/meanConst*
valueB
 *    *
dtype0
D
random_normal_26/stddevConst*
valueB
 *  �?*
dtype0
�
%random_normal_26/RandomStandardNormalRandomStandardNormalrandom_normal_26/shape*
T0*
dtype0*
seed2 *

seed 
d
random_normal_26/mulMul%random_normal_26/RandomStandardNormalrandom_normal_26/stddev*
T0
M
random_normal_26Addrandom_normal_26/mulrandom_normal_26/mean*
T0
�
Conv2D_6Conv2DblockOutputLayer_5random_normal_26*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
D
random_normal_27/shapeConst*
valueB: *
dtype0
B
random_normal_27/meanConst*
valueB
 *    *
dtype0
D
random_normal_27/stddevConst*
valueB
 *  �?*
dtype0
�
%random_normal_27/RandomStandardNormalRandomStandardNormalrandom_normal_27/shape*
T0*
dtype0*
seed2 *

seed 
d
random_normal_27/mulMul%random_normal_27/RandomStandardNormalrandom_normal_27/stddev*
T0
M
random_normal_27Addrandom_normal_27/mulrandom_normal_27/mean*
T0
Q

BiasAdd_13BiasAddConv2D_6random_normal_27*
data_formatNHWC*
T0
*
blockLayer1_6Relu
BiasAdd_13*
T0
S
random_normal_28/shapeConst*
dtype0*%
valueB"              
B
random_normal_28/meanConst*
dtype0*
valueB
 *    
D
random_normal_28/stddevConst*
valueB
 *  �?*
dtype0
�
%random_normal_28/RandomStandardNormalRandomStandardNormalrandom_normal_28/shape*
T0*
dtype0*
seed2 *

seed 
d
random_normal_28/mulMul%random_normal_28/RandomStandardNormalrandom_normal_28/stddev*
T0
M
random_normal_28Addrandom_normal_28/mulrandom_normal_28/mean*
T0
�
blockLayer2_6Conv2DblockLayer1_6random_normal_28*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
D
random_normal_29/shapeConst*
dtype0*
valueB: 
B
random_normal_29/meanConst*
valueB
 *    *
dtype0
D
random_normal_29/stddevConst*
valueB
 *  �?*
dtype0
�
%random_normal_29/RandomStandardNormalRandomStandardNormalrandom_normal_29/shape*
dtype0*
seed2 *

seed *
T0
d
random_normal_29/mulMul%random_normal_29/RandomStandardNormalrandom_normal_29/stddev*
T0
M
random_normal_29Addrandom_normal_29/mulrandom_normal_29/mean*
T0
V

BiasAdd_14BiasAddblockLayer2_6random_normal_29*
T0*
data_formatNHWC
2
Cast_6/xConst*
value	B
 Z *
dtype0

;
blockLayer3_6/SwitchSwitchCast_6/xCast_6/x*
T0

C
blockLayer3_6/switch_tIdentityblockLayer3_6/Switch:1*
T0

A
blockLayer3_6/switch_fIdentityblockLayer3_6/Switch*
T0

4
blockLayer3_6/pred_idIdentityCast_6/x*
T0

w
!blockLayer3_6/random_normal/shapeConst^blockLayer3_6/switch_t*%
valueB"              *
dtype0
f
 blockLayer3_6/random_normal/meanConst^blockLayer3_6/switch_t*
valueB
 *    *
dtype0
h
"blockLayer3_6/random_normal/stddevConst^blockLayer3_6/switch_t*
valueB
 *  �?*
dtype0
�
0blockLayer3_6/random_normal/RandomStandardNormalRandomStandardNormal!blockLayer3_6/random_normal/shape*
T0*
dtype0*
seed2 *

seed 
�
blockLayer3_6/random_normal/mulMul0blockLayer3_6/random_normal/RandomStandardNormal"blockLayer3_6/random_normal/stddev*
T0
n
blockLayer3_6/random_normalAddblockLayer3_6/random_normal/mul blockLayer3_6/random_normal/mean*
T0
�
blockLayer3_6/Conv2DConv2DblockLayer3_6/Conv2D/Switch:1blockLayer3_6/random_normal*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
blockLayer3_6/Conv2D/SwitchSwitchblockOutputLayer_5blockLayer3_6/pred_id*
T0*%
_class
loc:@blockOutputLayer_5
j
#blockLayer3_6/random_normal_1/shapeConst^blockLayer3_6/switch_t*
valueB: *
dtype0
h
"blockLayer3_6/random_normal_1/meanConst^blockLayer3_6/switch_t*
valueB
 *    *
dtype0
j
$blockLayer3_6/random_normal_1/stddevConst^blockLayer3_6/switch_t*
valueB
 *  �?*
dtype0
�
2blockLayer3_6/random_normal_1/RandomStandardNormalRandomStandardNormal#blockLayer3_6/random_normal_1/shape*
seed2 *

seed *
T0*
dtype0
�
!blockLayer3_6/random_normal_1/mulMul2blockLayer3_6/random_normal_1/RandomStandardNormal$blockLayer3_6/random_normal_1/stddev*
T0
t
blockLayer3_6/random_normal_1Add!blockLayer3_6/random_normal_1/mul"blockLayer3_6/random_normal_1/mean*
T0
u
blockLayer3_6/BiasAddBiasAddblockLayer3_6/Conv2DblockLayer3_6/random_normal_1*
T0*
data_formatNHWC
T
blockLayer3_6/AddAddblockLayer3_6/Add/Switch:1blockLayer3_6/BiasAdd*
T0
m
blockLayer3_6/Add/SwitchSwitch
BiasAdd_14blockLayer3_6/pred_id*
T0*
_class
loc:@BiasAdd_14
k
blockLayer3_6/Switch_1Switch
BiasAdd_14blockLayer3_6/pred_id*
T0*
_class
loc:@BiasAdd_14
Y
blockLayer3_6/MergeMergeblockLayer3_6/Switch_1blockLayer3_6/Add*
T0*
N
8
blockOutputLayer_6RelublockLayer3_6/Merge*
T0
�

maxpool1_1MaxPoolblockOutputLayer_6*
data_formatNHWC*
strides
*
ksize
 *
paddingSAME*
T0
B
Reshape/shapeConst*
valueB"����   *
dtype0
D
ReshapeReshape
maxpool1_1Reshape/shape*
Tshape0*
T0
K
random_normal_30/shapeConst*
valueB"      *
dtype0
B
random_normal_30/meanConst*
dtype0*
valueB
 *    
D
random_normal_30/stddevConst*
valueB
 *  �?*
dtype0
�
%random_normal_30/RandomStandardNormalRandomStandardNormalrandom_normal_30/shape*

seed *
T0*
dtype0*
seed2 
d
random_normal_30/mulMul%random_normal_30/RandomStandardNormalrandom_normal_30/stddev*
T0
M
random_normal_30Addrandom_normal_30/mulrandom_normal_30/mean*
T0
Z
MatMulMatMulReshaperandom_normal_30*
transpose_a( *
transpose_b( *
T0
D
random_normal_31/shapeConst*
valueB:*
dtype0
B
random_normal_31/meanConst*
valueB
 *    *
dtype0
D
random_normal_31/stddevConst*
valueB
 *  �?*
dtype0
�
%random_normal_31/RandomStandardNormalRandomStandardNormalrandom_normal_31/shape*
dtype0*
seed2 *

seed *
T0
d
random_normal_31/mulMul%random_normal_31/RandomStandardNormalrandom_normal_31/stddev*
T0
M
random_normal_31Addrandom_normal_31/mulrandom_normal_31/mean*
T0
L
outputsBiasAddMatMulrandom_normal_31*
T0*
data_formatNHWC"