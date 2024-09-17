#======================================================================
# deterministic bandit experiments
#======================================================================
env_names_list=('all')
algo_name_list=('pg_update')
initial_policy_list=('uniform' 'bad')
exp_name=('constant_eta')

t=(1000000)
num_instances=(10)
functional_update=('False')
eta_list=(100 1000 10000 100000 1000000)
num_arms_list=(2 5 10)

parallel -j 65 \
    OMP_NUM_THREADS=1 \
    python pg_experiments.py ::: \
    --env_names ::: ${env_names_list[@]} ::: \
    --algo_names ::: ${algo_name_list[@]} ::: \
    --initial_policy ::: ${initial_policy_list[@]} ::: \
    --exp_name ::: ${exp_name[@]} ::: \
    --t ::: ${t[@]} ::: \
    --num_instances ::: ${num_instances[@]} ::: \
    --functional_update ::: ${functional_update[@]} ::: \
    --eta ::: ${eta_list[@]} ::: \
    --num_arms ::: ${num_arms_list[@]} ::: \