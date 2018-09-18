#!/usr/bin/env bash
## The first things in your script needs to be the PBS configuration for Torque,
## anything after is your code. PBS directives are denoted by a line starting with
## #PBS then the option you want to configure. If the line starts with anything else
## for example: ##PBS, it is ignored. This is why there are many ## here.
##
## Once a _SINGLE_ command is issued then torque will no long look for
## further #PBS commands in the file.
##
## Job submission: qsub <scriptfile>
## Job status: qstat -a
## Delete a job you enqueued: qdel <job id>
##
## 1 nodes, 4 CPUs, wall clock time of 1 hours, 4 GB of memory, 2 GPUs
## There is ONLY 1 node available, so that should be 1 always however ppn (Processors
## per Node) can be between 1 and 32, the amount available on this system.
## Walltime is the amount of time to allow your program to run before it is killed
## --> Walltime can be excluded for infinite running time allowed, gpus if excluded will mean 0 gpus requested
##
##PBS -l walltime=4:00:00,mem=32gb,nodes=1:ppn=10:gpus=2

#PBS -l walltime=72:00:00,mem=50gb,nodes=1:ppn=7

## example with only 4 processors
##   -> PBS -l nodes=1:ppn=4
## example with just 2 gpus and 2 processors
##   -> PBS -l nodes=1:ppn=2:gpus=2
## example with 2 cpus and 1 hour of processing time
##   -> PBS -l nodes=1:ppn=2,walltime=1:00:00
##
## Submit job to the no-limits queue (only available to members of "compnet" group)
#PBS -q high
##
## merge error and output to single file
#PBS -j oe
##
## Rename the output file
##PBS -o ex.o
##
## Rename the error file
##PBS -e example.sh.err
##
## Only run the job after the specified time, date is omittable as well as seconds
## however the hour and minut specifier need to be stated
##PBS -a [[[[CC]YY]MM]DD]hhmm.[SS]
##
## Specifies the arguments to pass to the script, valid options are like
##   qsub -F "myarg1 myarg2 myarg3=myarg3value"
##PBS -F "myarg1"
##
## send mail if the process aborts, when it begins, and
## when it ends (abe)
##PBS -m abe
##PBS -M <your username, e.g. sean.lawlor>
##
## Specifies custom environment variables to set for the script
##PBS -v TESTVAR="test value"
##
## Specifies the desired shell for this job
##PBS -S /bin/bash
##
## Job dependency: Specifies that a specific job must complete prior to this job.
## The following only shows if the job completed OK, otherwise see: FMI see:
## http://docs.adaptivecomputing.com/torque/5-0-1/Content/topics/torque/commands/qsub.htm
##PBS -W depend=afterok:<jobid>[:<jobid2>:<jobid3>:...]

######## Variables available to Torque jobs ##################
## PBS_JOBNAME	User specified jobname
## PBS_ARRAYID	Zero-based value of job array index for this job (in version 2.2.0 and later)
## PBS_GPUFILE	Line-delimited list of GPUs allocated to the job located in $TORQUE_HOME/aux/jobidgpu. Each line follows the following format:
##		<host>-gpu<number> For example, myhost-gpu1.
## PBS_O_WORKDIR	Job's submission directory
## PBS_TASKNUM	Number of tasks requested
## PBS_O_HOME	Home directory of submitting user
## PBS_MOMPORT	Active port for MOM daemon
## PBS_O_LOGNAME	Name of submitting user
## REM PBS_O_LANG	Language variable for job
## PBS_JOBCOOKIE	Job cookie
## PBS_JOBID	Unique pbs job id
## PBS_NODENUM	Node offset number
## PBS_NUM_NODES	Number of nodes allocated to the job
## PBS_NUM_PPN	Number of procs per node allocated to the job
## PBS_O_SHELL	Script shell
## PBS_O_HOST	Host on which job script is currently running
## PBS_QUEUE	Job queue
## PBS_NODEFILE	File containing line delimited list of nodes allocated to the job
## PBS_NP		Number of execution slots (cores) for the job
## PBS_O_PATH	Path variable used to locate executables within job script

PBS_O_WORKDIR="/home/floregol/gcn/gcn/greedy_sampling/"
cd $PBS_O_WORKDIR
source activate gcn
export CUDA_VISIBLE_DEVICES="0"
python reproduce_greedy.py
