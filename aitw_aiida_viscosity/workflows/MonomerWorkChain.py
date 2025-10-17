"""Implementation of the WorkChain for AITW viscosity calculation."""
import copy

from aiida import orm
from aiida.common import exceptions as exc
from aiida.engine import ToContext, WorkChain, if_, while_
from aiida.plugins import SchedulerFactory
from aiida_shell import launch_shell_job

from . import functions as fnc

BASENAME = 'aiida'
DIRECT_SCHEDULER = SchedulerFactory('core.direct')

def clean_calcjob_remote(node):
    """Clean the remote directory of a ``CalcJobNode``."""
    cleaned = False
    try:
        node.outputs.remote_folder._clean()  # pylint: disable=protected-access
        cleaned = True
    except (IOError, OSError, KeyError):
        pass
    return cleaned

def clean_workchain_calcs(workchain):
    """Clean all remote directories of a workchain's descendant calculations."""
    cleaned_calcs = []

    for called_descendant in workchain.called_descendants:
        if isinstance(called_descendant, orm.CalcJobNode):
            if clean_calcjob_remote(called_descendant):
                cleaned_calcs.append(called_descendant.pk)

    return cleaned_calcs

def validate_deform_velocities(node: orm.List, _):
    """Validate that all deformation velocities are positive."""
    for value in node.get_list():
        if value <= 0.0:
            return 'All deformation velocities must be positive.'

class MonomerWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        # INPUTS ############################################################################
        spec.input('num_steps', valid_type=orm.Int, help='The number of MD steps to run in the NEMD simulation.')
        spec.input('smiles_string', valid_type=orm.Str, help='The SMILES string of the molecule to simulate.')
        spec.input(
            'reference_temperature', valid_type=orm.Float,
            help='The reference temperature in Kelvin for the simulation.'
        )
        spec.input(
            'force_field', valid_type=orm.Str,
            default=lambda: orm.Str('gaff2'),
            help='The SMILES string of the molecule to simulate.'
        )
        spec.input(
            'nmols', valid_type=orm.Int,
            default=lambda: orm.Int(1000),
            help='The number of molecules to insert into the simulation box.'
        )
        spec.input(
            'time_step', valid_type=orm.Float,
            default=lambda: orm.Float(0.001),
            help='The MD time step in picoseconds.'
        )
        spec.input(
            'deform_velocities', valid_type=orm.List,
            default=lambda: orm.List(list=[0.005, 0.002, 0.05, 0.02, 0.01, 0.1, 0.2]),
            validator=validate_deform_velocities,
            help=(
                'List of deformation velocities to use in the NEMD simulations. '
                'See https://manual.gromacs.org/current/user-guide/mdp-options.html#mdp-deform for details.'
            )
        )

        spec.input(
            'veloxchem_basis', valid_type=orm.Str,
            default=lambda: orm.Str('6-31G*'),
            help=(
                'The basis set to use in the VeloxChem calculation. This should be 6-31G* for RESP partial charges '
                'with the GAFF and GAFF2 force fields. '
                'See https://veloxchem.org/docs/basis_sets.html for details and available basis sets.'
            )
        )
        spec.input(
            'gromacs_minimization_steps', valid_type=orm.Int,
            default=lambda: orm.Int(5000),
            help='The number of steps to use in the GROMACS minimization.'
        )
        spec.input(
            'gromacs_equilibration_steps', valid_type=orm.Int,
            default=lambda: orm.Int(500000),
            help='The number of steps to use in the GROMACS equilibration.'
        )

        spec.input('acpype_code', valid_type=orm.AbstractCode, help='Code for running the `acpype` program.')
        spec.input('obabel_code', valid_type=orm.AbstractCode, help='Code for running the `obabel` program.')
        spec.input('veloxchem_code', valid_type=orm.AbstractCode, help='Code for python with `veloxchem` installed.')
        spec.input('gmx_code', valid_type=orm.AbstractCode, help='Code for running `gmx` or `gmx_mpi`.')
        spec.input(
            'gmx_code_local', valid_type=orm.AbstractCode,
            required=False,
            help='Code for running `gmx` or `gmx_mpi` locally for initialization/serial runs.'
        )

        spec.input(
            'with_mpi', valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help='If `True`, run calculations with MPI when possible.'
        )
        spec.input(
            'max_num_machines', valid_type=orm.Int,
            default=lambda: orm.Int(1),
            help='The maximum number of machines (nodes) to use for the calculations.'
        )
        spec.input(
            'max_wallclock_seconds', valid_type=orm.Int,
            default=lambda: orm.Int(3600),
            help='The maximum wallclock time in seconds for the calculations.'
        )
        spec.input(
            'clean_workdir', valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.'
        )

        # OUTLINE ############################################################################
        spec.outline(
            cls.setup,

            cls.submit_acpype,
            cls.inspect_acpype,

            cls.submit_obabel,
            cls.inspect_obabel,

            cls.make_veloxchem_input,
            cls.submit_veloxchem,
            cls.inspect_veloxchem,

            cls.run_resp_injection,
            cls.update_top_file,
            cls.get_box_size,

            cls.submit_insertmol,
            cls.inspect_insertmol,

            cls.make_gromacs_minimization_input,
            cls.submit_minimization_init,
            cls.submit_minimization_run,
            cls.inspect_minimization,

            cls.make_gromacs_equilibration_input,
            cls.submit_equilibration_init,
            cls.submit_equilibration_run,
            cls.inspect_equilibration,

            cls.extract_equilibrated_box_length,

            cls.make_nemd_inputs,
            cls.submit_nemd_init,
            if_(cls.should_do_alltogheter)(
                cls.submit_nemd_run_parallel,
            ).else_(
                while_(cls.do_nemd_serial)(
                    cls.submit_nemd_run_serial,
                )
            ),
            cls.inspect_nemd,

            cls.submit_energy_parallel,
            cls.inspect_energy,

            cls.collect_pressure_averages,
            cls.compute_viscosity_data,
            cls.fit_viscosity,
        )

        # OUTPUTS ############################################################################
        spec.output(
            'equilibrated_box_length_nm',
            valid_type=orm.Float,
            help='The edge length of the equilibrated cubic simulation box in nanometers.'
        )
        spec.output(
            'viscosity_data',
            valid_type=orm.ArrayData,
            help=(
                'ArrayData containing `deformation_velocities`, `pressure_averages`, `shear_rates`, '
                'and `viscosities` arrays.'
            )
        )
        spec.output(
            'xyz', valid_type=orm.SinglefileData,
            help='The XYZ file of the molecule generated from ACPYPE bi the SMILES code.'
        )
        spec.output(
            'system_gro', valid_type=orm.SinglefileData,
            help='The .gro file of the full simulation box after inserting all molecules.'
        )
        spec.output(
            'equilibrated_gro', valid_type=orm.SinglefileData,
            help='The equilibrated .gro file after minimization and equilibration.'
        )

        spec.output_namespace(
            'acpype',
            valid_type=orm.SinglefileData,
            dynamic=True,
            help='ACPYPE output files.'
        )
        spec.output_namespace(
            'nemd',
            valid_type=orm.SinglefileData,
            dynamic=True,
            help='NEMD .edr output files for each deformation velocity.'
        )
        spec.output(
            'eta_N', valid_type=orm.Float, required=False,
            help='TODO description of quantity'
        )
        spec.output(
            'sigma_E', valid_type=orm.Float, required=False,
            help='TODO description of quantity'
        )

        # ERRORS ############################################################################
        spec.exit_code(
            300, 'ERROR_ACPYPE_FAILED',
            message='The ACPYPE calculation did not finish successfully.'
        )
        spec.exit_code(
            305, 'ERROR_ACPYPE_MISSING_OUTPUT',
            message='ACPYPE did not produce all expected output files.'
        )
        spec.exit_code(
            310, 'ERROR_OBABEL_FAILED',
            message='The Open Babel calculation did not finish successfully.'
        )
        spec.exit_code(
            315, 'ERROR_OBELAB_MISSING_OUTPUT',
            message='Open Babel did not produce the expected output file.'
        )
        spec.exit_code(
            320, 'ERROR_VELOXCHEM_FAILED',
            message='The VeloxChem calculation did not finish successfully.'
        )
        spec.exit_code(
            325, 'ERROR_VELOXCHEM_MISSING_OUTPUT',
            message='VeloxChem did not produce the expected output file.'
        )
        spec.exit_code(
            330, 'ERROR_SUB_PROCESS_FAILED_GMX_MISSING_OUTPUT',
            message='A GROMACS minimization subprocess calculation failed.'
        )
        spec.exit_code(
            335, 'ERROR_SUB_PROCESS_FAILED_GMX_GROMP',
            message='A GROMACS grompp subprocess calculation failed.'
        )
        spec.exit_code(
            340, 'ERROR_SUB_PROCESS_FAILED_GMX_INSERTMOL',
            message='A GROMACS insert-molecules subprocess calculation failed.'
        )
        spec.exit_code(
            341, 'ERROR_SUB_PROCESS_FAILED_GMX_MINIMIZATION',
            message='A GROMACS minimization subprocess calculation failed.'
        )
        spec.exit_code(
            342, 'ERROR_SUB_PROCESS_FAILED_GMX_EQUILIBRATION',
            message='A GROMACS equilibration subprocess calculation failed.'
        )
        spec.exit_code(
            343, 'ERROR_SUB_PROCESS_FAILED_GMX_NEMD',
            message='A GROMACS NEMD subprocess calculation failed.'
        )
        spec.exit_code(
            344, 'ERROR_SUB_PROCESS_FAILED_GMX_ENERGY',
            message='A GROMACS energy subprocess calculation failed.'
        )

    def setup(self):
        """Setup context variables."""
        # Use remote code if local code not provided
        gmx_remote = self.inputs.gmx_code
        gmx_local = self.inputs.gmx_code_local if 'gmx_code_local' in self.inputs else gmx_remote
        self.report(f'Using GROMACS <{gmx_remote.pk}> for remote execution.')
        self.ctx.gmx_code_local = gmx_local
        self.report(f'Using GROMACS <{gmx_local.pk}> for local execution.')

        self.ctx.gmx_code_local = gmx_local

        self.ctx.smiles_string = self.inputs.smiles_string.value
        self.ctx.ff = self.inputs.force_field.value
        self.ctx.nmols = self.inputs.nmols.value

        gmx_computer: orm.Computer = self.inputs.gmx_code.computer
        gmx_sched = gmx_computer.get_scheduler()
        self.ctx.gromacs_serial_metadata = {
            'options': {
                'withmpi': False,
                'resources': {
                    'num_machines': 1,
                    'num_mpiprocs_per_machine': 1,
                },
                'max_wallclock_seconds': self.inputs.max_wallclock_seconds.value,
                'redirect_stderr': True,
            }
        }
        self.ctx.gmx_run_metadata = ptr = {
            'options': {
                'withmpi': self.inputs.with_mpi.value,
                'resources': {
                    'num_machines': self.inputs.max_num_machines.value,
                    'num_mpiprocs_per_machine': gmx_computer.get_default_mpiprocs_per_machine(),
                },
                'max_wallclock_seconds': self.inputs.max_wallclock_seconds.value,
                'redirect_stderr': True,
            }
        }

        max_mem = gmx_computer.get_default_memory_per_machine()
        if max_mem is not None:
            ptr['options']['max_memory_kb'] = max_mem

        self.ctx.gmx_computer = gmx_computer
        self.ctx.gmx_scheduler = gmx_sched

    def submit_acpype(self):
        """Submit acpype and obabel calculations to generate initial structure and parameters."""
        self.report('Running acpype through aiida-shell...')
        _, node = launch_shell_job(
            self.inputs.acpype_code,
            arguments=f'-i {{smiles}} -n 0 -c bcc -q sqm -b {BASENAME} -a {{ff}} -s 108000',
            nodes={
                'smiles': self.inputs.smiles_string,
                'ff': self.inputs.force_field
            },
            metadata={
                'call_link_label': 'acpype',
                'options': {
                    'withmpi': False,
                },
            },
            outputs=[f'{BASENAME}.acpype'],
            submit=True
        )
        self.report(f'Submitted job: {node}')
        # self.report(f'Outputs: {results_acpype}')
        return ToContext(acpype_calc=node)

    def inspect_acpype(self):
        """Inspect the output of the ACPYPE calculation and store relevant files in the context."""
        # Expected filename patterns
        target_suffixes = {
            'gro': 'GMX.gro',
            'itp': 'GMX.itp',
            'top': 'GMX.top',
            'pdb': 'NEW.pdb',
        }

        calc = self.ctx.acpype_calc
        if not calc.is_finished_ok:
            self.report('ACPYPE calculation failed.')
            return self.exit_codes.ERROR_ACPYPE_FAILED
        res = calc.outputs

        missing = []
        files = {}
        folder = res[f'{BASENAME}_acpype']
        for key, suffix in target_suffixes.items():
            try:
                file = fnc.extract_files_suffix(folder, orm.Str(suffix).store())
            except ValueError:
                missing.append(key)
                continue
            files[key] = file
            setattr(self.ctx, key, file)
        if missing:
            self.report(f'Missing expected output files from ACPYPE: {missing}')
            return self.exit_codes.ERROR_ACPYPE_MISSING_OUTPUT
        self.out('acpype', files)

    def submit_obabel(self):
        """Convert PDB file to XYZ using Open Babel."""
        self.report('Running obabel through aiida-shell...')
        out_filename = f'{BASENAME}.xyz'
        _, node = launch_shell_job(
            self.inputs.obabel_code,
            arguments = f'{{pdbfile}} -O {out_filename}',
            nodes={
                'pdbfile': self.ctx.pdb
            },
            metadata={
                'call_link_label': 'obabel',
                'options': {
                    'withmpi': False,
                    'redirect_stderr': True,
                }
            },
            outputs=[out_filename],
            submit=True
        )

        self.report(f'Submitted job: {node}')

        return ToContext(obabel_calc=node)

    def inspect_obabel(self):
        """Inspect the output of the Open Babel calculation."""
        calc = self.ctx.obabel_calc
        if not calc.is_finished_ok:
            self.report('Open Babel calculation failed.')
            return self.exit_codes.ERROR_OBABEL_FAILED

        try:
            xyz_file = calc.outputs[f'{BASENAME}_xyz']
        except exc.NotExistentKeyError:
            self.report('Open Babel did not produce the expected XYZ output file.')
            return self.exit_codes.ERROR_OBELAB_MISSING_OUTPUT
        self.ctx.xyz = xyz_file
        self.out('xyz', xyz_file)

    def make_veloxchem_input(self):
        """Prepare input files for VeloxChem calculation."""
        self.ctx.veloxchem_input = fnc.generate_veloxchem_input(self.inputs.veloxchem_basis)

    def submit_veloxchem(self):
        """Submit a VeloxChem calculation to compute RESP charges and store the resulting PDB file"""
        self.report('Running veloxchem through aiida-shell...')
        _, node = launch_shell_job(
            self.inputs.veloxchem_code,
            arguments = '{script_file} {xyzfile}',
            nodes={
                'script_file': self.ctx.veloxchem_input,
                'xyzfile': self.ctx.xyz
            },
            metadata = {
                'call_link_label': 'veloxchem',
                'options': {
                    'withmpi': False,
                }
            },
            submit=True,
        )

        self.report(f'Submitted job: {node}')
        return ToContext(veloxchem_calc=node)

    def inspect_veloxchem(self):
        """Extract the PDB file with RESP charges from the VeloxChem calculation."""
        calc = self.ctx.veloxchem_calc
        if not calc.is_finished_ok:
            self.report('VeloxChem calculation failed.')
            return self.exit_codes.ERROR_VELOXCHEM_FAILED

        try:
            pdb_file = calc.outputs['stdout']
        except exc.NotExistentKeyError:
            self.report('VeloxChem did not produce the expected PDB output file.')
            return self.exit_codes.ERROR_VELOXCHEM_MISSING_OUTPUT
        self.ctx.pdb = pdb_file

    def run_resp_injection(self):
        """
        Inject RESP charges from a PDB file into an ITP file and store the result in self.ctx.
        Assumes self.ctx.pdb and self.ctx.itp are SinglefileData nodes.
        """
        self.ctx.itp_with_resp = fnc.run_resp_injection(
            pdb_file=self.ctx.pdb,
            itp_file=self.ctx.itp
        )

        self.report(f'Updated ITP file with RESP charges stored as node {self.ctx.itp_with_resp}')

    def update_top_file(self):
        """Update the .top file to reference the new .itp file and correct molecule count."""
        self.ctx.top_updated = fnc.update_top_file(
            nmols=self.inputs.nmols,
            top_file=self.ctx.top,
            itp_file=self.ctx.itp_with_resp,
        )

        self.report(f"Updated .top file stored: {self.ctx.top_updated.filename} {self.ctx.top_updated}")

    def get_box_size(self):
        """Approximate molar mass from SMILES string using RDKit and store as AiiDA Float node."""
        self.ctx.box_size = fnc.get_box_size(
            nmols=self.inputs.nmols,
            smiles_string=self.inputs.smiles_string
        )

        self.report(f"Calculated box edge length: {self.ctx.box_size.value:.2f} nm")

    def submit_insertmol(self):
        self.report(f'Running GROMACS insert-molecules to create a box of {self.inputs.nmols.value} molecules... ')
        filename = f'{BASENAME}.gro'
        metadata = copy.deepcopy(self.ctx.gromacs_serial_metadata)
        metadata['call_link_label'] = 'insert_molecules'
        _, node = launch_shell_job(
            self.ctx.gmx_code_local,
            arguments=(
                f'insert-molecules -ci {{grofile}} -o {filename} -nmol {{nmols}} ' +
                '-try 1000 -box {box_vector} {box_vector} {box_vector}'
            ),
            nodes={
                'grofile': self.ctx.gro,
                'nmols': self.inputs.nmols,
                'box_vector': self.ctx.box_size
            },
            metadata=metadata,
            outputs=[filename],
            submit=True
        )
        self.report(f'Submitted job: {node}')

        return ToContext(insertmol_calc=node)

    def inspect_insertmol(self):
        """Inspect the output of the insert-molecules calculation."""
        calc = self.ctx.insertmol_calc
        if not calc.is_finished_ok:
            self.report('GROMACS insert-molecules calculation failed.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_INSERTMOL

        try:
            system_gro = calc.outputs[f'{BASENAME}_gro']
        except exc.NotExistentKeyError:
            self.report('GROMACS insert-molecules did not produce the expected output file.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_MISSING_OUTPUT

        self.ctx.system_gro = system_gro
        self.out('system_gro', system_gro)

    def make_gromacs_minimization_input(self):
        """Generate a basic GROMACS minimization input file."""
        self.ctx.minimize_mdp = fnc.generate_gromacs_minimization_input(
            minimization_steps=self.inputs.gromacs_minimization_steps
        )

    def submit_minimization_init(self):
        """Initialize GROMACS minimization run to generate .tpr file."""
        self.report('Running GROMACS minimization initialization...')
        metadata = copy.deepcopy(self.ctx.gromacs_serial_metadata)
        metadata['call_link_label'] = 'minimization_grompp'
        _, node = launch_shell_job(
            self.ctx.gmx_code_local,
            arguments='grompp -f {mdpfile} -c {grofile} -r {grofile} -p {topfile} -o minimize.tpr',
            nodes={
                'mdpfile': self.ctx.minimize_mdp,
                'grofile': self.ctx.system_gro,
                'topfile': self.ctx.top_updated,
                'itpfile': self.ctx.itp_with_resp
            },
            metadata=metadata,
            outputs=['mdout.mdp', 'minimize.tpr'],
            submit=True
        )
        self.report(f'Submitted job: {node}')

        return ToContext(gromp_minimize_calc=node)

    def submit_minimization_run(self):
        """Run GROMACS minimization mdrun."""
        gromp_calc = self.ctx.gromp_minimize_calc
        if not gromp_calc.is_finished_ok:
            self.report('GROMACS grompp for minimization failed.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_GROMP
        try:
            tpr_file = gromp_calc.outputs['minimize_tpr']
        except exc.NotExistentKeyError:
            self.report('GROMP for minimization failed')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_MISSING_OUTPUT

        self.report('Running GROMACS minimization mdrun...')
        metadata = copy.deepcopy(self.ctx.gmx_run_metadata)
        metadata['call_link_label'] = 'minimization_mdrun'
        # gmx_mpi mdrun -v -deffnm minimize
        _, node = launch_shell_job(
            self.inputs.gmx_code,
            arguments='mdrun -v -s {tprfile} -deffnm minimize',
            nodes={
                'tprfile': tpr_file
            },
            metadata=metadata,
            outputs=['minimize.gro'],
            submit=True
        )
        self.report(f'Submitted job: {node}')

        return ToContext(minimize_calc=node)

    def inspect_minimization(self):
        """Inspect the output of the minimization calculation."""
        calc = self.ctx.minimize_calc
        if not calc.is_finished_ok:
            self.report('GROMACS minimization calculation failed.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_MINIMIZATION

        try:
            minimized_gro = calc.outputs['minimize_gro']
        except exc.NotExistentKeyError:
            self.report('GROMACS minimization did not produce the expected output file.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_MISSING_OUTPUT

        self.ctx.minimized_gro = minimized_gro

    def make_gromacs_equilibration_input(self):
        """Generate a basic GROMACS equilibration input file."""
        self.ctx.equilibrate_mdp = fnc.generate_gromacs_equilibration_input(
            num_steps=self.inputs.gromacs_equilibration_steps,
            time_step=self.inputs.time_step,
            reference_temperature=self.inputs.reference_temperature
        )

    def submit_equilibration_init(self):
        """Initialize GROMACS equilibration run to generate .tpr file."""
        self.report('Running GROMACS equilibration run INIT...')
        metadata = copy.deepcopy(self.ctx.gromacs_serial_metadata)
        metadata['call_link_label'] = 'equilibration_grompp'
        out_filename = 'equilibrate.tpr'
        _, node = launch_shell_job(
            self.ctx.gmx_code_local,
            arguments='grompp -f {mdpfile} -c {grofile} -r {grofile} -p {topfile} -o ' + out_filename,
            nodes={
                'mdpfile': self.ctx.equilibrate_mdp,
                'grofile': self.ctx.minimized_gro,
                'topfile': self.ctx.top_updated,
                'itpfile': self.ctx.itp_with_resp
            },
            metadata=metadata,
            outputs=['mdout.mdp', out_filename],
            submit=True
        )
        self.report(f'Submitted job: {node}')

        return ToContext(gromp_equilibrate_calc=node)

    def submit_equilibration_run(self):
        """Run GROMACS equilibration mdrun."""
        calc = self.ctx.gromp_equilibrate_calc
        if not calc.is_finished_ok:
            self.report('GROMACS grompp for equilibration failed.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_GROMP
        try:
            tpr_file = calc.outputs['equilibrate_tpr']
        except exc.NotExistentKeyError:
            self.report('GROMP for equilibration failed')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_MISSING_OUTPUT

        self.report('Running GROMACS equilibration run MDRUN...')
        metadata = copy.deepcopy(self.ctx.gmx_run_metadata)
        metadata['call_link_label'] = 'equilibration_mdrun'
        out_filename = 'equilibrate.gro'
        _, node = launch_shell_job(
            self.inputs.gmx_code,
            arguments='mdrun -v -s {tprfile} -deffnm equilibrate',
            nodes={
                'tprfile': tpr_file
            },
            metadata=metadata,
            outputs=[out_filename],
            submit=True
        )
        self.report(f'Submitted job: {node}')

        return ToContext(equilibrate_calc=node)

    def inspect_equilibration(self):
        """Inspect the output of the equilibration calculation."""
        calc = self.ctx.equilibrate_calc
        if not calc.is_finished_ok:
            self.report('GROMACS equilibration calculation failed.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_EQUILIBRATION
        try:
            equilibrated_gro = calc.outputs['equilibrate_gro']
        except exc.NotExistentKeyError:
            self.report('GROMACS equilibration did not produce the expected output file.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_MISSING_OUTPUT

        self.ctx.equilibrated_gro = equilibrated_gro
        self.out('equilibrated_gro', equilibrated_gro)

    def make_nemd_inputs(self):
        """Prepare input files for NEMD simulations with different deformation velocities."""
        self.report('Preparing NEMD input files for each deformation velocity...')
        self.report(str(self.inputs.deform_velocities))

        self.ctx.str_defvel = {defvel: fnc.string_safe_float(defvel) for defvel in self.inputs.deform_velocities}

        self.ctx.mdp_files = fnc.generate_gromacs_deform_vel_inputs(
            nsteps=self.inputs.num_steps,
            time_step=self.inputs.time_step,
            ref_t=self.inputs.reference_temperature,
            deform_velocities=self.inputs.deform_velocities
        )

    def submit_nemd_init(self):
        """Submit GROMACS grompp for each deformation velocity to generate .tpr files."""
        self.report('Running GROMACS NEMD initialization for each deformation velocity...')
        metadata = copy.deepcopy(self.ctx.gromacs_serial_metadata)
        metadata['call_link_label'] = 'nemd_grompp'
        fname = 'aiida.tpr'
        for defvel, str_defvel in self.ctx.str_defvel.items():
            mdp_file = self.ctx.mdp_files[f'mdp_{str_defvel}']
            _, node = launch_shell_job(
                self.ctx.gmx_code_local,
                arguments='grompp -f {mdpfile} -c {grofile} -r {grofile} -p {topfile} -o ' + fname,
                nodes={
                    'mdpfile': mdp_file,
                    'grofile': self.ctx.equilibrated_gro,
                    'topfile': self.ctx.top_updated,
                    'itpfile': self.ctx.itp_with_resp,
                },
                metadata=metadata,
                outputs=[fname],
                submit=True
            )
            self.report(f'Submitted job for deformation velocity {defvel}: {node}')
            self.to_context(**{f'grompp_{str_defvel}': node})

    def should_do_alltogheter(self) -> bool:
        """Check if all deformation velocity can be in parallel runs."""
        sched = self.ctx.gmx_scheduler
        if isinstance(sched, DIRECT_SCHEDULER):
            self.report('Direct scheduler does not support running multiple jobs.')
            self.ctx.nemd_serial_cnt = 0
            return False
        return True

    def submit_nemd_run_parallel(self):
        """Submit all NEMD runs as parallel/concurrent jobs."""
        self.report('Submitting GROMACS NEMD runs as parallel jobs...')
        metadata = copy.deepcopy(self.ctx.gmx_run_metadata)
        metadata['call_link_label'] = 'nemd_mdrun'
        for defvel, str_defvel in self.ctx.str_defvel.items():
            tpr_calc = self.ctx[f'grompp_{str_defvel}']
            if not tpr_calc.is_finished_ok:
                self.report(f'GROMACS grompp for deformation velocity {defvel} failed.')
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_GROMP
            try:
                tpr_file = tpr_calc.outputs['aiida_tpr']
            except exc.NotExistentKeyError:
                self.report(f'GROMP for deformation velocity {defvel} failed')
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_MISSING_OUTPUT

            _, node = launch_shell_job(
                self.inputs.gmx_code,
                arguments='mdrun -v -s {tpr_file} -deffnm ' + BASENAME,
                nodes={
                    'tpr_file': tpr_file,
                },
                metadata=metadata,
                outputs=[f'{BASENAME}.edr'],
                submit=True
            )

            self.report(f'Submitted job: {node}')
            self.to_context(**{f'nemd_{str_defvel}': node})

    def do_nemd_serial(self) -> bool:
        """Check if there are remaining deformation velocity to run in serial."""
        return self.ctx.nemd_serial_cnt < len(self.inputs.deform_velocities)

    def submit_nemd_run_serial(self):
        """Submit the next NEMD run in serial."""
        defvel = self.inputs.deform_velocities[self.ctx.nemd_serial_cnt]
        str_defvel = self.ctx.str_defvel[defvel]
        self.ctx.nemd_serial_cnt += 1

        tpr_calc = self.ctx[f'grompp_{str_defvel}']
        if not tpr_calc.is_finished_ok:
            self.report(f'GROMACS grompp for deformation velocity {defvel} failed.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_GROMP
        try:
            tpr_file = tpr_calc.outputs['aiida_tpr']
        except exc.NotExistentKeyError:
            self.report(f'GROMP for deformation velocity {defvel} failed')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_MISSING_OUTPUT

        self.report(f'Submitting GROMACS NEMD run for deformation velocity {defvel} as a serial job...')
        metadata = copy.deepcopy(self.ctx.gmx_run_metadata)
        metadata['call_link_label'] = 'nemd_mdrun'
        _, node = launch_shell_job(
            self.inputs.gmx_code,
            arguments='mdrun -v -s {tpr_file} -deffnm ' + BASENAME,
            nodes={
                'tpr_file': tpr_file,
            },
            metadata=metadata,
            outputs=[f'{BASENAME}.edr'],
            submit=True
        )

        self.report(f'Submitted job: {node}')

        return ToContext(**{f'nemd_{str_defvel}': node})

    def inspect_nemd(self):
        """Collect .edr files from the NEMD parallel run."""
        self.report('Collecting .edr files from NEMD runs...')

        calc_map = {defvel: self.ctx[f'nemd_{str_defvel}'] for defvel, str_defvel in self.ctx.str_defvel.items()}
        failed = [defvel for defvel, calc in calc_map.items() if not calc.is_finished_ok]
        if failed:
            self.report(f'NEMD runs for deformation velocities {failed} did not finish successfully.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_NEMD

        edr_files = {}
        edr_outputs = {}
        self.ctx.edr_outputs = {}
        for defvel, calc in calc_map.items():
            str_defvel = self.ctx.str_defvel[defvel]
            edr_file = calc.outputs['aiida_edr']
            edr_files[defvel] = edr_file
            edr_outputs[f'edr_{str_defvel}'] = edr_file
            self.report(f'Collected .edr file for deformation velocity {defvel}')

        self.ctx.edr_files = edr_files
        self.out('nemd', edr_outputs)

    def submit_energy_parallel(self):
        """Run `gmx energy` to extract pressure data from each EDR file."""
        self.report('Submitting GROMACS energy extraction runs for each deformation velocity...')
        self.ctx.pressure_xvg = {}
        for defvel, edr_file in self.ctx.edr_files.items():
            str_defvel = self.ctx.str_defvel[defvel]
            _, node = launch_shell_job(
                self.ctx.gmx_code_local,
                arguments=f'energy -f {{edr}} -o {BASENAME}.xvg',
                nodes={
                    'edr': edr_file,
                    # Select term 38, confirm with 0
                    'stdin': orm.SinglefileData.from_string('38\n0\n', filename='stdin'),
                },
                outputs=[f'{BASENAME}.xvg'],
                metadata={
                    'call_link_label': f'gromacs_energy',
                    'options': {
                        'resources': {'num_machines': 1},
                        'withmpi': False,
                        'redirect_stderr': True,
                        'filename_stdin': 'stdin'
                    }
                },
                submit=True
            )

            self.report(f'Submitted job: {node}')
            self.to_context(**{f'energy_{str_defvel}': node})

    def inspect_energy(self):
        """Collect .xvg files from the energy extraction runs."""
        calc_map = {defvel: self.ctx[f'energy_{str_defvel}'] for defvel, str_defvel in self.ctx.str_defvel.items()}

        failed = [defvel for defvel, calc in calc_map.items() if not calc.is_finished_ok]
        if failed:
            self.report(f'Energy extraction runs for deformation velocities {failed} did not finish successfully.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_ENERGY

        self.ctx.pressure_xvg = {}
        for defvel, calc in calc_map.items():
            xvg_file = calc.outputs['aiida_xvg']
            self.ctx.pressure_xvg[defvel] = xvg_file

    def extract_equilibrated_box_length(self):
        """Extract box length from the equilibrated .gro file."""
        box_length_nm = fnc.extract_box_length(self.ctx.equilibrated_gro)
        self.report(f"Box length extracted: {box_length_nm.value} nm")
        self.ctx.box_length_nm = box_length_nm
        self.out('equilibrated_box_length_nm', box_length_nm)

    def collect_pressure_averages(self):
        """Collect average pressures from the postprocessing workchain."""
        pressures = {}
        has_negatives = False
        for defvel in self.inputs.deform_velocities:
            str_defvel = self.ctx.str_defvel[defvel]
            xvg_file = self.ctx.pressure_xvg[defvel]
            avg_pressure = fnc.extract_pressure_from_xvg(xvg_file)
            pressures[f'pressure_{str_defvel}'] = avg_pressure
            if avg_pressure.value < 0:
                has_negatives = True
            self.report(f"Average pressure for deformation velocity {defvel}: {avg_pressure.value} bar")

        if has_negatives:
            self.report('WARNING: Negative pressures detected, results may be unphysical!')

        self.ctx.pressures = fnc.join_pressure_results(self.inputs.deform_velocities, **pressures)

    def compute_viscosity_data(self):
        """Compute average pressures, shear rates, and viscosities."""
        res = fnc.compute_viscosities(
            deformation_velocities=self.inputs.deform_velocities,
            pressures=self.ctx.pressures,
            box_length=self.ctx.box_length_nm
        )

        self.ctx.viscosity_data = res
        self.out('viscosity_data', res)

    def fit_viscosity(self):
        """Fit viscosity data to the Eyring model."""
        try:
            dct = fnc.fit_viscosity(self.ctx.viscosity_data)
        except ValueError:
            self.report('Fitting viscosity data to Eyring model failed.')
        else:
            eta_N = dct['eta_N']
            sigma_E = dct['sigma_E']
            self.out('eta_N', eta_N)
            self.out('sigma_E', sigma_E)
            self.report(
                f'Fitted viscosity data to Eyring model: eta_N={eta_N.value:.3e}, sigma_E={sigma_E.value:.3e}'
            )

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = clean_workchain_calcs(self.node)

        if cleaned_calcs:
            self.report(f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}")
