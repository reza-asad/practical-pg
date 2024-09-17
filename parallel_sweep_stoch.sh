#======================================================================
# stochastic bandit experiments
#======================================================================
env_names_list=('all')
algo_name_list=('spg_update')
initial_policy_list=('uniform' 'bad')
exp_name=('constant_eta')

t=(1000000)
num_instances=(50)
functional_update=('False')
eta_list=(0.001 0.01 0.1 1 10)
num_arms_list=(2 5 10)

parallel -j 55 \
    OMP_NUM_THREADS=1 \
    python spg_experiments.py ::: \
    --env_names ::: ${env_names_list[@]} ::: \
    --algo_names ::: ${algo_name_list[@]} ::: \
    --initial_policy ::: ${initial_policy_list[@]} ::: \
    --exp_name ::: ${exp_name[@]} ::: \
    --t ::: ${t[@]} ::: \
    --num_instances ::: ${num_instances[@]} ::: \
    --functional_update ::: ${functional_update[@]} ::: \
    --eta ::: ${eta_list[@]} ::: \
    --num_arms ::: ${num_arms_list[@]} ::: \