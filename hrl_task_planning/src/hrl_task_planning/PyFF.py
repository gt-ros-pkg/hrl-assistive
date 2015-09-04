#!/usr/bin/env python

from tempfile import NamedTemporaryFile
from subprocess import check_output, CalledProcessError
from os import remove

from pddl_utils import Planner


class FF(Planner):
    """ A solver instance based on an FF executable. """
    def __init__(self, problem, domain_file, ff_executable='./ff'):
        super(FF, self).__init__(problem, domain_file)
        self.ff_executable = ff_executable

    def _parse_solution(self, soln_txt):
        """ Extract list of solution steps from FF output. """
        sol = []
        soln_txt = soln_txt.split('step')[1].strip()
        soln_txt = soln_txt.split('time spent')[0].strip()
        steps = [step.strip() for step in soln_txt.splitlines()]
        for step in steps:
            args = step.split(':')[1].lstrip().split()
            act = args.pop(0)  # Remove action, leave all args
            sol.append({'act': act, 'args': args})
        return sol

    def solve(self):
        """ Create a temporary problem file and call FF to solve. """
        with NamedTemporaryFile() as problem_file:
            self.problem.to_file(self.problem, problem_file.name)
            try:
                soln_txt = check_output([self.ff_executable, '-o', self.domain_file, '-f', problem_file.name])
            except CalledProcessError as cpe:
                if "goal can be simplified to TRUE." in cpe.output:
                    self.solution = []
                    return self.solution
                else:
                    print "Warning: FF Could not find a solution to problem: %s in domain %s" % (self.problem.name,
                                                                                                 self.domain_file)
                    print self.problem
                    print "Output: ", cpe.output
                    self.solution = False
                    return self.solution
            finally:
                # clean up the soln file produced by ff (avoids large dumps of files in /tmp)
                try:
                    remove('.'.join([problem_file.name, 'soln']))
                except OSError as ose:
                    if ose.errno != 2:
                        raise ose
        self.solution = self._parse_solution(soln_txt)
        return self.solution
