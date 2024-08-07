# segin
```
git clone https://github.com/haddocking/segin.git
```
#### This steerable equivariant GNN is based on the framework of DeepRank-GNN 
#### steerable features: relative displacement is represented using spherical harmonics 
#### relative displacement between interacted residue i and residue j: 
$$ d_{ij} = \begin{bmatrix}
{x_i} - {x_j} \\
{y_i} - {y_j} \\
{z_i} - {z_j}
\end{bmatrix}
$$
![image](https://github.com/user-attachments/assets/c041606b-a119-4d93-9bd8-7156cc775bde)


The libraries used are the same as DeepRank-GNN-esm, with an extra e3nn library.  

```
conda env create -f deeprank.yml
```
https://github.com/e3nn/e3nn  
https://github.com/RobDHess/Steerable-E3-GNN

