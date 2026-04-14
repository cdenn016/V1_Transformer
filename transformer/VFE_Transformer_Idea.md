# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:34:43 2026

@author: chris and christine
"""
A Principled Gauge–VFE Transformer / LLM
Executive summary

The clean version of a gauge–VFE language model is:

Each token position carries a belief, not a single hidden vector:

𝑞
𝑖
(
𝑧
)
=
𝑁
(
𝜇
𝑖
,
Σ
𝑖
)
,
𝑔
𝑖
=
exp
⁡
(
𝜙
𝑖
 ⁣
⋅
 ⁣
𝐺
)
∈
𝐺
.
q
i
	​

(z)=N(μ
i
	​

,Σ
i
	​

),g
i
	​

=exp(ϕ
i
	​

⋅G)∈G.

Attention is geometric:

𝛽
𝑖
𝑗
∝
exp
⁡
 ⁣
(
−
1
𝜅
 
𝐷
 ⁣
(
𝑞
𝑖
 
∥
 
Ω
𝑖
𝑗
[
𝑞
𝑗
]
)
)
,
Ω
𝑖
𝑗
=
𝑔
𝑖
𝑔
𝑗
−
1
.
β
ij
	​

∝exp(−
κ
1
	​

D(q
i
	​

∥Ω
ij
	​

[q
j
	​

])),Ω
ij
	​

=g
i
	​

g
j
−1
	​

.
Each block performs an E-step on 
(
𝜇
,
Σ
,
𝜙
)
(μ,Σ,ϕ) by minimizing a variational free energy built from prior consistency and gauge-transported alignment.
The M-step is ordinary supervised language modeling:
solve for 
𝑞
⋆
q
⋆
 from the context only, then minimize cross-entropy of the next-token prediction produced from 
𝑞
⋆
q
⋆
.
Do not feed the target token into the E-step during LM training.
In the current code, use_obs_in_vfe=True literally passes targets into the E-step path, which is target leakage and creates a train/test mismatch.
Use the same statistical manifold for encode, inference, and decode.
The cleanest implementation is a PriorBank that serves as both encoder and decoder: tokens map to priors 
𝜋
𝑣
π
v
	​

, and final logits are produced by KL-to-prior decoding. The current code already supports this design.
1. Latent state and gauge action

At sequence position 
𝑖
i, let the latent token state be

𝑞
𝑖
(
𝑧
)
=
𝑁
(
𝑧
;
𝜇
𝑖
,
Σ
𝑖
)
,
𝜇
𝑖
∈
𝑅
𝐾
,
  
Σ
𝑖
∈
S
P
D
(
𝐾
)
,
q
i
	​

(z)=N(z;μ
i
	​

,Σ
i
	​

),μ
i
	​

∈R
K
,Σ
i
	​

∈SPD(K),

together with a gauge frame coordinate

𝜙
𝑖
∈
𝑔
,
𝑔
𝑖
=
exp
⁡
(
𝜙
𝑖
⋅
𝐺
)
∈
𝐺
.
ϕ
i
	​

∈g,g
i
	​

=exp(ϕ
i
	​

⋅G)∈G.

A local gauge transformation 
ℎ
𝑖
∈
𝐺
h
i
	​

∈G acts by

𝜇
𝑖
↦
ℎ
𝑖
𝜇
𝑖
,
Σ
𝑖
↦
ℎ
𝑖
Σ
𝑖
ℎ
𝑖
⊤
,
𝑔
𝑖
↦
ℎ
𝑖
𝑔
𝑖
.
μ
i
	​

↦h
i
	​

μ
i
	​

,Σ
i
	​

↦h
i
	​

Σ
i
	​

h
i
⊤
	​

,g
i
	​

↦h
i
	​

g
i
	​

.

The induced transport from 
𝑗
j to 
𝑖
i is

Ω
𝑖
𝑗
=
𝑔
𝑖
𝑔
𝑗
−
1
.
Ω
ij
	​

=g
i
	​

g
j
−1
	​

.

The transported Gaussian is therefore

Ω
𝑖
𝑗
[
𝑞
𝑗
]
=
𝑁
 ⁣
(
Ω
𝑖
𝑗
𝜇
𝑗
,
  
Ω
𝑖
𝑗
Σ
𝑗
Ω
𝑖
𝑗
⊤
)
.
Ω
ij
	​

[q
j
	​

]=N(Ω
ij
	​

μ
j
	​

,Ω
ij
	​

Σ
j
	​

Ω
ij
⊤
	​

).

That covariance sandwich product is the central correctness invariant for the whole framework and is explicitly treated that way in the transport code.

2. Vocabulary as a prior bank

Let the vocabulary define a family of token priors

Π
=
{
𝜋
𝑣
}
𝑣
=
1
𝑉
,
𝜋
𝑣
(
𝑧
)
=
𝑁
(
𝑧
;
𝜇
𝑣
𝜋
,
Σ
𝑣
𝜋
)
.
Π={π
v
	​

}
v=1
V
	​

,π
v
	​

(z)=N(z;μ
v
π
	​

,Σ
v
π
	​

).

The most principled parameterization is the gauge-fixed orbit form

𝜋
𝑣
=
𝐴
𝑣
▹
𝜋
0
,
𝐴
𝑣
=
exp
⁡
(
𝜓
𝑣
⋅
𝐺
)
,
π
v
	​

=A
v
	​

▹π
0
	​

,A
v
	​

=exp(ψ
v
	​

⋅G),

so that

𝜇
𝑣
𝜋
=
𝐴
𝑣
𝜇
0
,
Σ
𝑣
𝜋
=
𝐴
𝑣
Σ
0
𝐴
𝑣
⊤
.
μ
v
π
	​

=A
v
	​

μ
0
	​

,Σ
v
π
	​

=A
v
	​

Σ
0
	​

A
v
⊤
	​

.

This is already the intended meaning of the current PriorBank: it is a unified token-dependent prior bank that can serve as both embedding and output projection, with encode by prior lookup and decode by KL-to-prior logits.

Encoding

For input token 
𝑥
𝑖
x
i
	​

,

(
𝜇
𝑖
(
0
)
,
Σ
𝑖
(
0
)
,
𝜙
𝑖
(
0
)
)
=
E
n
c
o
d
e
(
𝑥
𝑖
)
.
(μ
i
(0)
	​

,Σ
i
(0)
	​

,ϕ
i
(0)
	​

)=Encode(x
i
	​

).

A natural implementation is

𝜇
𝑖
(
0
)
=
𝜇
𝑥
𝑖
𝜋
,
Σ
𝑖
(
0
)
=
Σ
𝑥
𝑖
𝜋
,
𝜙
𝑖
(
0
)
=
𝜓
𝑥
𝑖
.
μ
i
(0)
	​

=μ
x
i
	​

π
	​

,Σ
i
(0)
	​

=Σ
x
i
	​

π
	​

,ϕ
i
(0)
	​

=ψ
x
i
	​

	​

.
Decoding

Given a final latent belief 
𝑞
𝑖
⋆
q
i
⋆
	​

, define token logits by

ℓ
𝑖
,
𝑣
=
−
1
𝜏
d
e
c
 
𝐷
 ⁣
(
𝑞
𝑖
⋆
 
∥
 
𝜋
𝑣
)
,
ℓ
i,v
	​

=−
τ
dec
	​

1
	​

D(q
i
⋆
	​

∥π
v
	​

),

and

𝑝
𝜃
(
𝑦
𝑖
=
𝑣
∣
𝑥
<
𝑖
)
=
exp
⁡
(
ℓ
𝑖
,
𝑣
)
∑
𝑢
=
1
𝑉
exp
⁡
(
ℓ
𝑖
,
𝑢
)
.
p
θ
	​

(y
i
	​

=v∣x
<i
	​

)=
∑
u=1
V
	​

exp(ℓ
i,u
	​

)
exp(ℓ
i,v
	​

)
	​

.

This is exactly the right readout if you want the model’s hidden state, priors, and observation model to all live on the same Gaussian statistical manifold. The current PriorBank.decode() does this, though by default it uses a diagonal approximation unless full_cov_decode=True.

3. Positional structure as gauge composition

Position should enter as a contribution to the gauge frame, not merely as an additive Euclidean feature.

Let 
𝑝
𝑖
∈
𝑔
p
i
	​

∈g be a learned or fixed positional Lie-algebra element. Then set

𝜙
𝑖
(
0
)
=
B
C
H
 ⁣
(
𝜙
𝑖
token
,
 
𝑝
𝑖
)
,
ϕ
i
(0)
	​

=BCH(ϕ
i
token
	​

,p
i
	​

),

equivalently

𝑔
𝑖
(
0
)
=
exp
⁡
(
𝜙
𝑖
token
 ⁣
⋅
 ⁣
𝐺
)
 
exp
⁡
(
𝑝
𝑖
 ⁣
⋅
 ⁣
𝐺
)
.
g
i
(0)
	​

=exp(ϕ
i
token
	​

⋅G)exp(p
i
	​

⋅G).

This makes transport

Ω
𝑖
𝑗
=
𝑔
𝑖
𝑔
𝑗
−
1
Ω
ij
	​

=g
i
	​

g
j
−1
	​


depend on relative position in a gauge-covariant way. The current model is already designed around this interpretation: positional gauge frames compose with token gauge frames using BCH/Lie-algebra composition, and transport then inherits relative position.

4. Gauge-KL attention

Define the pairwise geometric divergence at layer 
ℓ
ℓ:

𝐷
𝑖
𝑗
(
ℓ
)
=
𝐷
 ⁣
(
𝑞
𝑖
(
ℓ
)
 
∥
 
Ω
𝑖
𝑗
(
ℓ
)
[
𝑞
𝑗
(
ℓ
)
]
)
.
D
ij
(ℓ)
	​

=D(q
i
(ℓ)
	​

	​

Ω
ij
(ℓ)
	​

[q
j
(ℓ)
	​

]).

Then attention is

𝛽
𝑖
𝑗
(
ℓ
)
=
exp
⁡
 ⁣
(
−
𝐷
𝑖
𝑗
(
ℓ
)
/
𝜅
ℓ
)
 
𝑚
𝑖
𝑗
∑
𝑘
exp
⁡
 ⁣
(
−
𝐷
𝑖
𝑘
(
ℓ
)
/
𝜅
ℓ
)
 
𝑚
𝑖
𝑘
,
β
ij
(ℓ)
	​

=
∑
k
	​

exp(−D
ik
(ℓ)
	​

/κ
ℓ
	​

)m
ik
	​

exp(−D
ij
(ℓ)
	​

/κ
ℓ
	​

)m
ij
	​

	​

,

where 
𝑚
𝑖
𝑗
m
ij
	​

 is the causal mask.

This is the core architectural thesis of the current model: attention is KL-based on the statistical manifold with gauge transport, not produced by learned 
𝑊
𝑄
,
𝑊
𝐾
W
Q
	​

,W
K
	​

 projections.

A natural transported message mean is

𝜇
ˉ
𝑖
(
ℓ
)
=
∑
𝑗
𝛽
𝑖
𝑗
(
ℓ
)
 
Ω
𝑖
𝑗
(
ℓ
)
𝜇
𝑗
(
ℓ
)
.
μ
ˉ
	​

i
(ℓ)
	​

=
j
∑
	​

β
ij
(ℓ)
	​

Ω
ij
(ℓ)
	​

μ
j
(ℓ)
	​

.

For covariance, there are two principled choices.

4.1 Mixture / moment-matching aggregator
Σ
ˉ
𝑖
(
ℓ
)
=
∑
𝑗
𝛽
𝑖
𝑗
(
ℓ
)
[
Ω
𝑖
𝑗
Σ
𝑗
(
ℓ
)
Ω
𝑖
𝑗
⊤
+
(
Ω
𝑖
𝑗
𝜇
𝑗
(
ℓ
)
−
𝜇
ˉ
𝑖
(
ℓ
)
)
(
Ω
𝑖
𝑗
𝜇
𝑗
(
ℓ
)
−
𝜇
ˉ
𝑖
(
ℓ
)
)
⊤
]
.
Σ
ˉ
i
(ℓ)
	​

=
j
∑
	​

β
ij
(ℓ)
	​

[Ω
ij
	​

Σ
j
(ℓ)
	​

Ω
ij
⊤
	​

+(Ω
ij
	​

μ
j
(ℓ)
	​

−
μ
ˉ
	​

i
(ℓ)
	​

)(Ω
ij
	​

μ
j
(ℓ)
	​

−
μ
ˉ
	​

i
(ℓ)
	​

)
⊤
].
4.2 Precision / product-of-experts aggregator

Define natural parameters

Λ
𝑖
𝑗
(
ℓ
)
=
(
Ω
𝑖
𝑗
Σ
𝑗
(
ℓ
)
Ω
𝑖
𝑗
⊤
)
−
1
,
𝜂
𝑖
𝑗
(
ℓ
)
=
Λ
𝑖
𝑗
(
ℓ
)
 
Ω
𝑖
𝑗
𝜇
𝑗
(
ℓ
)
.
Λ
ij
(ℓ)
	​

=(Ω
ij
	​

Σ
j
(ℓ)
	​

Ω
ij
⊤
	​

)
−1
,η
ij
(ℓ)
	​

=Λ
ij
(ℓ)
	​

Ω
ij
	​

μ
j
(ℓ)
	​

.

Then

Λ
ˉ
𝑖
(
ℓ
)
=
∑
𝑗
𝛽
𝑖
𝑗
(
ℓ
)
Λ
𝑖
𝑗
(
ℓ
)
,
𝜂
ˉ
𝑖
(
ℓ
)
=
∑
𝑗
𝛽
𝑖
𝑗
(
ℓ
)
𝜂
𝑖
𝑗
(
ℓ
)
,
Λ
ˉ
i
(ℓ)
	​

=
j
∑
	​

β
ij
(ℓ)
	​

Λ
ij
(ℓ)
	​

,
η
ˉ
	​

i
(ℓ)
	​

=
j
∑
	​

β
ij
(ℓ)
	​

η
ij
(ℓ)
	​

,
Σ
ˉ
𝑖
(
ℓ
)
=
(
Λ
ˉ
𝑖
(
ℓ
)
)
−
1
,
𝜇
ˉ
𝑖
(
ℓ
)
=
Σ
ˉ
𝑖
(
ℓ
)
𝜂
ˉ
𝑖
(
ℓ
)
.
Σ
ˉ
i
(ℓ)
	​

=(
Λ
ˉ
i
(ℓ)
	​

)
−1
,
μ
ˉ
	​

i
(ℓ)
	​

=
Σ
ˉ
i
(ℓ)
	​

η
ˉ
	​

i
(ℓ)
	​

.

If one wants “precision aggregation,” this is the mathematically coherent version: mean and covariance must both be fused in natural coordinates.

5. The per-layer E-step free energy

At layer 
ℓ
ℓ, let 
𝑝
𝑖
(
ℓ
)
p
i
(ℓ)
	​

 denote the layer prior and 
𝑞
𝑖
(
ℓ
)
q
i
(ℓ)
	​

 the current belief. A cleaned-up E-step free energy is

	
𝐹
ℓ
(
𝑞
,
𝜙
)
=
𝛼
ℓ
∑
𝑖
𝐷
 ⁣
(
𝑞
𝑖
(
ℓ
)
 
∥
 
𝑝
𝑖
(
ℓ
)
)
+
𝜆
a
l
i
g
n
,
ℓ
∑
𝑖
,
𝑗
𝛽
𝑖
𝑗
(
ℓ
)
𝐷
 ⁣
(
𝑞
𝑖
(
ℓ
)
 
∥
 
Ω
𝑖
𝑗
(
ℓ
)
[
𝑞
𝑗
(
ℓ
)
]
)
+
𝜆
s
o
f
t
,
ℓ
 
𝐶
ℓ
(
𝑞
,
𝜙
)
+
𝜆
r
e
g
,
ℓ
 
𝑅
ℓ
(
𝑞
,
𝜙
)
.
		
(1)
F
ℓ
	​

(q,ϕ)=α
ℓ
	​

i
∑
	​

D(q
i
(ℓ)
	​

∥p
i
(ℓ)
	​

)+λ
align,ℓ
	​

i,j
∑
	​

β
ij
(ℓ)
	​

D(q
i
(ℓ)
	​

∥Ω
ij
(ℓ)
	​

[q
j
(ℓ)
	​

])+λ
soft,ℓ
	​

C
ℓ
	​

(q,ϕ)+λ
reg,ℓ
	​

R
ℓ
	​

(q,ϕ).
(1)

Here:

the first term is prior consistency,
the second is belief alignment through gauge transport,
𝐶
ℓ
C
ℓ
	​

 is the softmax-coupling correction,
𝑅
ℓ
R
ℓ
	​

 collects target-free regularizers.

The current VFE FFN already implements the essential two-term decomposition of the alignment gradient:

∂
𝜃
∑
𝑗
𝛽
𝑖
𝑗
𝐾
𝐿
𝑖
𝑗
=
∑
𝑗
𝛽
𝑖
𝑗
 
∂
𝜃
𝐾
𝐿
𝑖
𝑗
+
∑
𝑗
𝐾
𝐿
𝑖
𝑗
 
∂
𝜃
𝛽
𝑖
𝑗
,
∂
θ
	​

j
∑
	​

β
ij
	​

KL
ij
	​

=
j
∑
	​

β
ij
	​

∂
θ
	​

KL
ij
	​

+
j
∑
	​

KL
ij
	​

∂
θ
	​

β
ij
	​

,

with the direct term interpreted as a Boltzmann-style gating and the second as a softmax-coupling correction.

Crucial exclusion for LM training

For ordinary autoregressive language modeling, Equation (1) must not contain the supervised next-token target as an observation inside the E-step.

The current implementation does exactly that when use_obs_in_vfe=True: compute_free_energy_loss() has a flag explicitly described as “Pass targets into VFE E-step,” and model.forward_with_attention() documents targets as “used as observations in E-step.”

That is label leakage, not principled predictive inference.

6. E-step updates

Let 
𝑡
t index the inner-loop iterations. Then:

Mean update
	
𝜇
𝑖
(
ℓ
,
𝑡
+
1
)
=
𝜇
𝑖
(
ℓ
,
𝑡
)
−
𝜂
𝜇
 
Σ
𝑖
(
ℓ
,
𝑡
)
∇
𝜇
𝑖
𝐹
ℓ
.
		
(2)
μ
i
(ℓ,t+1)
	​

=μ
i
(ℓ,t)
	​

−η
μ
	​

Σ
i
(ℓ,t)
	​

∇
μ
i
	​

	​

F
ℓ
	​

.
(2)

This is the Gaussian-location natural gradient.

Covariance update

For full covariance, use an SPD retraction:

	
Σ
𝑖
(
ℓ
,
𝑡
+
1
)
=
Exp
⁡
Σ
𝑖
(
ℓ
,
𝑡
)
(
−
𝜂
Σ
 
grad
⁡
Σ
𝑖
𝐹
ℓ
)
.
		
(3)
Σ
i
(ℓ,t+1)
	​

=Exp
Σ
i
(ℓ,t)
	​

	​

(−η
Σ
	​

grad
Σ
i
	​

	​

F
ℓ
	​

).
(3)

A practical affine-invariant form is

Σ
𝑖
(
ℓ
,
𝑡
+
1
)
=
Σ
𝑖
1
/
2
exp
⁡
 ⁣
(
−
𝜂
Σ
 
Σ
𝑖
1
/
2
∇
Σ
𝑖
𝐹
ℓ
Σ
𝑖
1
/
2
)
Σ
𝑖
1
/
2
.
Σ
i
(ℓ,t+1)
	​

=Σ
i
1/2
	​

exp(−η
Σ
	​

Σ
i
1/2
	​

∇
Σ
i
	​

	​

F
ℓ
	​

Σ
i
1/2
	​

)Σ
i
1/2
	​

.
Gauge-frame update

For Lie-algebra coordinates,

	
𝜙
𝑖
(
ℓ
,
𝑡
+
1
)
=
𝜙
𝑖
(
ℓ
,
𝑡
)
−
𝜂
𝜙
 
𝑀
𝜙
−
1
(
𝜙
𝑖
(
ℓ
,
𝑡
)
)
∇
𝜙
𝑖
𝐹
ℓ
,
		
(4)
ϕ
i
(ℓ,t+1)
	​

=ϕ
i
(ℓ,t)
	​

−η
ϕ
	​

M
ϕ
−1
	​

(ϕ
i
(ℓ,t)
	​

)∇
ϕ
i
	​

	​

F
ℓ
	​

,
(4)

where 
𝑀
𝜙
M
ϕ
	​

 is a natural metric on the Lie algebra, such as a Killing-form or pullback metric.

The current optimizer / preconditioner stack is already built around this viewpoint: 
𝜙
ϕ-updates are treated geometrically via Killing/pullback-style preconditioning, while 
𝜇
μ and 
𝜎
σ have Gaussian Fisher-based preconditioning.

7. Cross-layer hierarchical prior handoff

A deep model requires a rule that turns layer 
ℓ
ℓ’s posterior into layer 
ℓ
+
1
ℓ+1’s prior.

The cleanest exact version is

	
𝜇
𝑝
,
𝑖
(
ℓ
+
1
)
=
𝜇
𝑖
(
ℓ
)
⋆
,
Σ
𝑝
,
𝑖
(
ℓ
+
1
)
=
Σ
𝑖
(
ℓ
)
⋆
.
		
(5)
μ
p,i
(ℓ+1)
	​

=μ
i
(ℓ)⋆
	​

,Σ
p,i
(ℓ+1)
	​

=Σ
i
(ℓ)⋆
	​

.
(5)

A damped version is safer in practice:

𝜇
𝑝
,
𝑖
(
ℓ
+
1
)
=
(
1
−
𝜌
𝜇
)
 
𝜇
𝑝
,
𝑖
(
ℓ
)
+
𝜌
𝜇
 
𝜇
𝑖
(
ℓ
)
⋆
,
μ
p,i
(ℓ+1)
	​

=(1−ρ
μ
	​

)μ
p,i
(ℓ)
	​

+ρ
μ
	​

μ
i
(ℓ)⋆
	​

,
	
Σ
𝑝
,
𝑖
(
ℓ
+
1
)
=
(
1
−
𝜌
Σ
)
 
Σ
𝑝
,
𝑖
(
ℓ
)
+
𝜌
Σ
 
Σ
𝑖
(
ℓ
)
⋆
,
0
<
𝜌
𝜇
,
𝜌
Σ
≤
1.
		
(6)
Σ
p,i
(ℓ+1)
	​

=(1−ρ
Σ
	​

)Σ
p,i
(ℓ)
	​

+ρ
Σ
	​

Σ
i
(ℓ)⋆
	​

,0<ρ
μ
	​

,ρ
Σ
	​

≤1.
(6)

This is deliberately stricter than the current stack, which propagates posterior 
𝜇
μ upward but keeps sigma_prior frozen at the embedding value to prevent sigma cascade. That is a stabilization trick, not a full hierarchical posterior-to-prior update.

8. Gauge-equivariant normalization

After the final layer, normalize with the Mahalanobis norm:

	
𝜇
~
𝑖
=
𝜇
𝑖
⋆
𝐾
𝜇
𝑖
⋆
⊤
Σ
𝑖
⋆
−
1
𝜇
𝑖
⋆
+
𝜀
.
		
(7)
μ
~
	​

i
	​

=μ
i
⋆
	​

μ
i
⋆⊤
	​

Σ
i
⋆−1
	​

μ
i
⋆
	​

+ε
K
	​

	​

.
(7)

This is the correct gauge-equivariant analogue of RMS-style normalization, because

𝜇
⊤
Σ
−
1
𝜇
μ
⊤
Σ
−1
μ

is a gauge scalar. The current MahalanobisNorm implements exactly this idea and explicitly states the approximate key-norm-bias cancellation property in isotropic / shared-metric regimes.

9. Decode and language-model objective

Given final normalized belief 
𝑞
𝑖
⋆
=
𝑁
(
𝜇
~
𝑖
,
Σ
𝑖
⋆
)
q
i
⋆
	​

=N(
μ
~
	​

i
	​

,Σ
i
⋆
	​

), define logits by KL decode:

ℓ
𝑖
,
𝑣
=
−
1
𝜏
d
e
c
𝐷
 ⁣
(
𝑞
𝑖
⋆
 
∥
 
𝜋
𝑣
)
,
ℓ
i,v
	​

=−
τ
dec
	​

1
	​

D(q
i
⋆
	​

∥π
v
	​

),
	
𝑝
𝜃
(
𝑦
𝑖
=
𝑣
∣
𝑥
<
𝑖
)
=
softmax
⁡
𝑣
(
ℓ
𝑖
,
𝑣
)
.
		
(8)
p
θ
	​

(y
i
	​

=v∣x
<i
	​

)=softmax
v
	​

(ℓ
i,v
	​

).
(8)

Then the training objective is

	
𝐿
L
M
(
𝜃
)
=
∑
𝑖
−
log
⁡
𝑝
𝜃
(
𝑦
𝑖
∣
𝑞
𝑖
⋆
(
𝑥
<
𝑖
)
)
+
𝜆
h
y
p
e
r
𝐻
(
𝜃
)
,
		
(9)
L
LM
	​

(θ)=
i
∑
	​

−logp
θ
	​

(y
i
	​

∣q
i
⋆
	​

(x
<i
	​

))+λ
hyper
	​

H(θ),
(9)

where 
𝑞
𝑖
⋆
(
𝑥
<
𝑖
)
q
i
⋆
	​

(x
<i
	​

) is obtained from context-only inference.

This is the clean variational split:

E-step: infer 
𝑞
⋆
q
⋆
 from the context only,
M-step: update slow parameters to improve next-token prediction from 
𝑞
⋆
q
⋆
.

In contrast, the current use_obs_in_vfe=True path passes targets into the E-step, while test evaluation explicitly forces use_obs_in_vfe=False. That is precisely why train PPL can collapse while validation/test blows up.

10. Optional no-target active inference inside the E-step

If one wants active-inference flavor during training-time inference, only use target-free terms.

Let

𝑝
𝑖
(
𝑣
)
=
𝑝
𝜃
(
𝑣
∣
𝑞
𝑖
)
p
i
	​

(v)=p
θ
	​

(v∣q
i
	​

)

be the model’s own predictive distribution from its current latent belief.

Then one may add:

Pragmatic confidence
𝐹
p
r
a
g
=
𝜆
p
r
a
g
∑
𝑖
𝐻
[
𝑝
𝑖
]
.
F
prag
	​

=λ
prag
	​

i
∑
	​

H[p
i
	​

].
Epistemic value
𝐹
e
p
i
=
−
𝜆
e
p
i
∑
𝑖
𝐼
(
𝑧
𝑖
;
𝑦
𝑖
∣
𝑞
𝑖
)
.
F
epi
	​

=−λ
epi
	​

i
∑
	​

I(z
i
	​

;y
i
	​

∣q
i
	​

).

Then the E-step objective becomes

	
𝐹
ℓ
A
I
=
𝐹
ℓ
+
𝐹
p
r
a
g
+
𝐹
e
p
i
.
		
(10)
F
ℓ
AI
	​

=F
ℓ
	​

+F
prag
	​

+F
epi
	​

.
(10)

This is close to the intended use of the current active_inference.py module, which explicitly describes the pragmatic term as entropy minimization of the model’s own prediction “without target leak,” and the epistemic term as BALD-like mutual information over the belief.

11. Generation-time expected free energy

For genuine Friston-style active inference in an LLM, the action is the next token.

For each candidate next token 
𝑎
a, define

	
𝐺
𝑡
(
𝑎
)
=
𝐸
𝑞
(
𝑜
∣
𝑎
)
[
−
log
⁡
𝑝
\*
(
𝑜
)
]
⏟
risk
+
𝐸
𝑞
(
𝑧
∣
𝑎
)
[
𝐻
[
𝑝
(
𝑜
∣
𝑧
)
]
]
⏟
ambiguity
−
𝐼
𝑞
(
𝑧
;
𝑜
∣
𝑎
)
⏟
epistemic value
.
		
(11)
G
t
	​

(a)=
risk
E
q(o∣a)
	​

[−logp
\*
(o)]
	​

	​

+
ambiguity
E
q(z∣a)
	​

[H[p(o∣z)]]
	​

	​

−
epistemic value
I
q
	​

(z;o∣a)
	​

	​

.
(11)

Then choose actions via the policy posterior

	
𝑞
𝑡
(
𝑎
)
∝
exp
⁡
(
−
𝛾
 
𝐺
𝑡
(
𝑎
)
)
.
		
(12)
q
t
	​

(a)∝exp(−γG
t
	​

(a)).
(12)

That is the correct place for active inference in a language model: generation-time action selection, not target-conditioned E-step inference during supervised training. The current expected_free_energy.py module already adopts exactly this interpretation and even states that its teacher-forced training loss is only a surrogate, not genuine active inference.

12. The cleaned-up architecture

The principled architecture is therefore

𝑥
1
:
𝑁
  
→
PriorBank encode
  
{
(
𝜇
𝑖
(
0
)
,
Σ
𝑖
(
0
)
,
𝜙
𝑖
(
0
)
)
}
𝑖
=
1
𝑁
  
→
positional BCH
  
{
(
𝜇
𝑖
(
0
)
,
Σ
𝑖
(
0
)
,
𝜙
𝑖
(
0
)
)
}
𝑖
=
1
𝑁
x
1:N
	​

PriorBank encode
	​

{(μ
i
(0)
	​

,Σ
i
(0)
	​

,ϕ
i
(0)
	​

)}
i=1
N
	​

positional BCH
	​

{(μ
i
(0)
	​

,Σ
i
(0)
	​

,ϕ
i
(0)
	​

)}
i=1
N
	​

  
→
𝐿
 gauge–VFE blocks
  
{
(
𝜇
𝑖
(
𝐿
)
⋆
,
Σ
𝑖
(
𝐿
)
⋆
,
𝜙
𝑖
(
𝐿
)
⋆
)
}
𝑖
=
1
𝑁
  
→
MahalanobisNorm
  
{
(
𝜇
~
𝑖
,
Σ
𝑖
(
𝐿
)
⋆
)
}
𝑖
=
1
𝑁
  
→
PriorBank decode
  
𝑝
𝜃
(
𝑦
𝑖
∣
𝑥
<
𝑖
)
.
L gauge–VFE blocks
	​

{(μ
i
(L)⋆
	​

,Σ
i
(L)⋆
	​

,ϕ
i
(L)⋆
	​

)}
i=1
N
	​

MahalanobisNorm
	​

{(
μ
~
	​

i
	​

,Σ
i
(L)⋆
	​

)}
i=1
N
	​

PriorBank decode
	​

p
θ
	​

(y
i
	​

∣x
<i
	​

).

Each block performs:

compute 
𝛽
𝑖
𝑗
β
ij
	​

 from gauge-transported KL,
compute the E-step gradients of 
𝐹
ℓ
F
ℓ
	​

,
update 
(
𝜇
,
Σ
,
𝜙
)
(μ,Σ,ϕ),
pass the posterior onward as the next layer’s prior.
13. Three design laws
Law 1: inference must not see the answer key

For ordinary language-model training, the E-step may depend on the context and the model’s own predictions, but not on the supervised next token.

Law 2: transport must act on covariance by conjugation

Any gauge transport of Gaussians must satisfy

Σ
↦
Ω
Σ
Ω
⊤
.
Σ↦ΩΣΩ
⊤
.

This is non-negotiable. The transport code treats it as the central invariant.

Law 3: encode, infer, and decode on the same manifold

If tokens are represented as Gaussian priors, then inference should evolve Gaussian beliefs, and decoding should compare the final belief against Gaussian token priors by KL. The current PriorBank is already set up to support exactly that.

14. Practical implementation choices
Recommended first version

Use:

flat transport,
Lie-algebra 
𝜙
ϕ parameterization rather than free 
Ω
Ω,
block-diagonal full covariance per head,
PriorBank encode/decode,
Mahalanobis normalization,
context-only E-step,
CE-only M-step,
no target-conditioned observation term.
Recommended second version

Add:

damped posterior-to-prior transfer across layers,
no-target pragmatic / epistemic active-inference terms inside the E-step,
learned per-layer temperatures 
𝜅
ℓ
κ
ℓ
	​

, 
𝜏
d
e
c
τ
dec
	​

.
Recommended third version

Only then add:

non-flat edge connections,
generation-time EFE policy over candidate next tokens.
15. Relationship to the current code

The cleaned-up specification above is not a fantasy rewrite; it is a principled consolidation of mechanisms the current code already contains in partial form:

KL-based gauge-transport attention is already the attention mechanism.
The VFE block already performs iterative E-step belief evolution.
The model already supports positional gauge composition via 
𝜙
ϕ-space encoding.
PriorBank already provides the right encode/decode abstraction.
MahalanobisNorm already provides the right gauge-equivariant normalization.
The active-inference modules already distinguish between target-free E-step shaping and generation-time EFE policy.

What must be removed for principled LM training is the target-conditioned E-step path. The code currently exposes that path directly, and evaluation already disables it, which is why it causes catastrophic train/test inconsistency.

16. Final distilled statement

A principled gauge–VFE transformer is a deep amortized variational model in which:

token states are local Gaussian beliefs,
gauge frames define transport between token beliefs,
attention is the Gibbs kernel of gauge-transported divergence,
each layer performs an inner variational E-step on 
(
𝜇
,
Σ
,
𝜙
)
(μ,Σ,ϕ),
the vocabulary is a bank of Gaussian priors,
decoding is KL-to-prior softmax,
the M-step is ordinary next-token learning from context-only inferred beliefs,
active inference, when used canonically, operates at generation time over candidate next-token actions.

That is the version that is mathematically clean, architecturally coherent, and compatible with language-model training without leakage.