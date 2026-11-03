/* Minimal Windows-friendly driver for cuPDLPx (its own CLI needs getopt/libgen).
   Usage: cupdlpx_run [options] FILE.mps
     --no_presolve --no_active_set_boost --no_infeasibility_detection --verbose
     --eps_opt X --eps_feas X --iter_limit N --time_limit S --eval_freq N --result FILE
   Prints a one-line machine-readable summary at the end. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cupdlpx.h"
#include "mps_parser.h"

static const char *reason_str(termination_reason_t r)
{
    switch (r)
    {
    case TERMINATION_REASON_OPTIMAL: return "OPTIMAL";
    case TERMINATION_REASON_PRIMAL_INFEASIBLE: return "PRIMAL_INFEASIBLE";
    case TERMINATION_REASON_DUAL_INFEASIBLE: return "DUAL_INFEASIBLE";
    case TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED: return "INFEASIBLE_OR_UNBOUNDED";
    case TERMINATION_REASON_TIME_LIMIT: return "TIME_LIMIT";
    case TERMINATION_REASON_ITERATION_LIMIT: return "ITERATION_LIMIT";
    case TERMINATION_REASON_FEAS_POLISH_SUCCESS: return "FEAS_POLISH_SUCCESS";
    default: return "UNSPECIFIED";
    }
}

int main(int argc, char **argv)
{
    pdhg_parameters_t params;
    set_default_parameters(&params);
    const char *file = NULL;
    const char *result_path = NULL;

    for (int i = 1; i < argc; i++)
    {
        const char *a = argv[i];
#define NEXT (i + 1 < argc ? argv[++i] : (fprintf(stderr, "missing value for %s\n", a), exit(2), ""))
        if (!strcmp(a, "--no_presolve")) params.presolve = false;
        else if (!strcmp(a, "--no_active_set_boost")) params.active_set_boost = false;
        else if (!strcmp(a, "--no_infeasibility_detection")) params.infeasibility_detection = false;
        else if (!strcmp(a, "--verbose") || !strcmp(a, "-v")) params.verbose = true;
        else if (!strcmp(a, "--eps_opt")) params.termination_criteria.eps_optimal_relative = atof(NEXT);
        else if (!strcmp(a, "--eps_feas")) params.termination_criteria.eps_feasible_relative = atof(NEXT);
        else if (!strcmp(a, "--iter_limit")) params.termination_criteria.iteration_limit = atoi(NEXT);
        else if (!strcmp(a, "--time_limit")) params.termination_criteria.time_sec_limit = atof(NEXT);
        else if (!strcmp(a, "--eval_freq")) params.termination_evaluation_frequency = atoi(NEXT);
        else if (!strcmp(a, "--result")) result_path = NEXT;
        else if (a[0] == '-') { fprintf(stderr, "unknown option %s\n", a); return 2; }
        else file = a;
#undef NEXT
    }
    if (!file) { fprintf(stderr, "usage: cupdlpx_run [options] FILE.mps\n"); return 2; }

    char err[512] = {0};
    if (cupdlpx_validate_parameters(&params, err, sizeof err)) { fprintf(stderr, "bad params: %s\n", err); return 2; }

    cupdlpx_gpu_init_async();
    lp_problem_t *prob = read_mps_file(file);
    if (!prob) { fprintf(stderr, "failed to read %s\n", file); return 1; }

    cupdlpx_result_t *r = solve_lp_problem(prob, &params);
    if (!r) { fprintf(stderr, "solver failed\n"); lp_problem_free(prob); return 1; }

    /* cuPDLPx logs through its own C runtime with its own stdout buffer, so
       on a pipe its output and ours interleave. --result keeps this line whole. */
    FILE *out = stdout;
    if (result_path && !(out = fopen(result_path, "w"))) { perror(result_path); return 1; }
    fprintf(out, "RESULT file=%s status=%s iters=%d time=%.3f primal_obj=%.12e dual_obj=%.12e "
           "rel_primal=%.3e rel_dual=%.3e rel_gap=%.3e\n",
           file, reason_str(r->termination_reason), r->total_count, r->cumulative_time_sec,
           r->primal_objective_value, r->dual_objective_value, r->relative_primal_residual,
           r->relative_dual_residual, r->relative_objective_gap);
    if (out != stdout) fclose(out);

    cupdlpx_result_free(r);
    lp_problem_free(prob);
    return 0;
}
