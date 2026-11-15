"""Native WarpX inputs: identical child positions/weights; physical species q,m unchanged."""
def write_warpx(path, cfg, xyz, n, halfR, metal, max_grid=128):
    R = cfg['radius_m']; half = halfR*R
    QE=cfg['charge_C']; ME=cfg['electron_mass_kg']; MI=cfg['ion_mass_kg']
    diag_names = [f'p{i}' for i in range(len(cfg['probes_m']))]
    names = diag_names+['number','extrema','momentum','energy']+(['charge_eb'] if metal else [])
    lines = [f'stop_time = {cfg["stop_time_s"]:.17g}',
             'max_step = 100000', f'amr.n_cell = {n} {n} {n}',
             'amr.max_level = 0', f'amr.max_grid_size = {max_grid}', 'amr.blocking_factor = 8',
             'geometry.dims = 3', f'geometry.prob_lo = {-half} {-half} {-half}',
             f'geometry.prob_hi = {half} {half} {half}', 'warpx.cfl = 0.9',
             'warpx.verbose = 1', 'warpx.use_filter = 0', 'warpx.serialize_initial_conditions = 1',
             'warpx.synchronize_velocity_for_diagnostics = 0',
             'boundary.field_lo = pml pml pml', 'boundary.field_hi = pml pml pml',
             'boundary.particle_lo = absorbing absorbing absorbing',
             'boundary.particle_hi = absorbing absorbing absorbing',
             'boundary.particle_eb = absorbing', 'warpx.pml_ncell = 12',
             'warpx.do_pml_in_domain = 0', 'algo.maxwell_solver = ect' if metal else 'algo.maxwell_solver = yee',
             'algo.current_deposition = esirkepov', 'algo.particle_pusher = boris',
             'algo.field_gathering = energy-conserving', 'algo.particle_shape = 1',
             'particles.species_names = electrons ions', 'particles.E_ext_particle_init_style = parse_E_ext_particle_function',
             'particles.Ex_external_particle_function(x,y,z,t) = "0"',
             'particles.Ey_external_particle_function(x,y,z,t) = "0"',
             f'particles.Ez_external_particle_function(x,y,z,t) = "{cfg["force_equivalent_E_V_m"]:.17g}*sin(pi*t/{cfg["pulse_duration_s"]:.17g})^2*(t>0)*(t<{cfg["pulse_duration_s"]:.17g})"',
             'diagnostics.diags_names = snapshots', 'snapshots.diag_type = Full',
             'snapshots.intervals = 100000', 'snapshots.fields_to_plot = Ex Ey Ez Bx By Bz rho',
             'snapshots.file_prefix = snapshots/plt',
             'warpx.reduced_diags_names = '+' '.join(names),
             'number.type = ParticleNumber', 'extrema.type = ParticleExtrema', 'extrema.species = electrons',
             'momentum.type = ParticleMomentum', 'energy.type = ParticleEnergy']
    if metal:
        lines += [f'warpx.eb_implicit_function = "{R*R:.17g}-x*x-y*y-z*z"', 'charge_eb.type = ChargeOnEB']
    for name in names:
        lines += [f'{name}.intervals = 1', f'{name}.path = reduced/']
    for i, point in enumerate(cfg['probes_m']):
        name = f'p{i}'
        lines += [f'{name}.type = FieldProbe', f'{name}.interp_order = 1']
        lines += [f'{name}.{axis}_probe = {value:.17g}' for axis,value in zip('xyz',point)]
    for name, charge, mass in [('electrons',-QE,ME),('ions',QE,MI)]:
        lines += [f'{name}.charge = {charge:.17g}', f'{name}.mass = {mass:.17g}',
                  f'{name}.injection_style = MultipleParticles']
        for i,axis in enumerate('xyz'):
            lines += [f'{name}.multiple_particles_pos_{axis} = '+' '.join(f'{x:.17g}' for x in xyz[:,i]),
                      f'{name}.multiple_particles_u{axis} = '+' '.join(['0']*len(xyz))]
        lines += [f'{name}.multiple_particles_weight = '+' '.join([f'{cfg["macro_weight"]:.17g}']*len(xyz))]
    # AMReX input records have a finite line length. Explicit particle lists
    # must use its documented backslash continuation syntax.
    wrapped=[]
    for line in lines:
        if len(line)>3000:
            key,value=line.split('=',1); tokens=value.split()
            chunks=[' '.join(tokens[i:i+48]) for i in range(0,len(tokens),48)]
            for i,chunk in enumerate(chunks):
                wrapped.append((key+'= ' if i==0 else '  ')+chunk+(' '+chr(92) if i<len(chunks)-1 else ''))
        else:
            wrapped.append(line)
    with path.open('w',encoding='utf-8',newline='\n') as stream:
        stream.write('\n'.join(wrapped)+'\n')
