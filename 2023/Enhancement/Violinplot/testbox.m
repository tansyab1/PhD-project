figure
niqe_mat_bm3d = transpose(niqe_mat_bm3d);
niqe_mat_cycleisp = transpose(niqe_mat_cycleisp);
niqe_mat_mir = transpose(niqe_mat_mir);
niqe_mat_uf = transpose(niqe_mat_uf);
niqe_mat_da = transpose(niqe_mat_da);
boxplot([niqe_mat_bm3d,niqe_mat_cycleisp,niqe_mat_da, niqe_mat_mir, niqe_mat_uf], 'notch', 'off')
title('Compare Random Data from Different Distributions')

