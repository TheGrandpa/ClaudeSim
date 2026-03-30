from dataclasses import dataclass, field


@dataclass
class SimConfig:
    # ── World ─────────────────────────────────────────────────────────────────
    world_width: float = 4000.0
    world_height: float = 4000.0

    # ── Resources ─────────────────────────────────────────────────────────────
    max_resources: int = 2000
    initial_resources: int = 800
    resource_spawn_rate: float = 3.0        # new resources per tick
    resource_min_energy: float = 10.0
    resource_max_energy: float = 40.0
    resource_radius: float = 6.0            # visual + eat detection radius

    # ── Creatures ─────────────────────────────────────────────────────────────
    initial_population: int = 60
    max_population: int = 300
    population_floor: int = 40             # inject randoms if below this
    population_floor_enabled: bool = True
    creature_radius: float = 16.0

    # ── Energy ────────────────────────────────────────────────────────────────
    initial_energy: float = 150.0
    max_energy: float = 500.0
    metabolic_cost_per_tick: float = 0.010  # base metabolism
    nn_cost_per_weight: float = 0.0002      # thinking cost scales with brain size
    movement_cost_factor: float = 0.01      # cost = factor * speed^2
    eat_cost: float = 0.5
    reproduce_cost: float = 60.0            # each parent pays this
    attack_cost: float = 2.0
    attack_damage: float = 15.0
    carnivore_efficiency: float = 0.55   # fraction of damage converted to energy for pure carnivore
    reproduce_energy_threshold: float = 110.0

    # ── Senses ────────────────────────────────────────────────────────────────
    sense_radius: float = 150.0             # general proximity sense range
    ray_count: int = 6
    ray_length: float = 200.0
    nearest_food_count: int = 3
    nearest_creature_count: int = 3

    # ── Physics ───────────────────────────────────────────────────────────────
    max_speed: float = 4.25
    max_turn_rate: float = 0.15             # radians per tick
    velocity_damping: float = 0.85

    # Stamina physics
    stamina_sprint_drain: float = 0.8   # stamina drained per tick at thrust=1 (scales with size)
    stamina_speed_floor:  float = 0.15  # minimum speed fraction when fully exhausted

    # ── NEAT ──────────────────────────────────────────────────────────────────
    neat_input_size: int = 59   # 24 rays + 9 food + 18 creatures(×6) + 2 hunger-dir + 5 self + 1 stamina
    neat_output_size: int = 7   # thrust, turn, eat, reproduce, attack, flee, signal
    weight_init_range: float = 2.0
    weight_perturb_rate: float = 0.8        # probability each weight is perturbed
    weight_perturb_strength: float = 0.3    # gaussian std dev
    weight_scramble_rate: float = 0.1       # probability weight fully reset
    add_connection_rate: float = 0.12       # per reproduction event
    add_node_rate: float = 0.06
    disable_connection_rate: float = 0.01
    max_hidden_nodes: int = 64
    allow_recurrent: bool = True           # enable recurrent (back-edge) connections
    recurrent_connection_rate: float = 0.3 # fraction of add_connection calls that may be recurrent

    # ── Speciation ────────────────────────────────────────────────────────────
    species_compatibility_threshold: float = 0.8   # distance to form a new species
    c1: float = 2.0                         # excess gene coefficient
    c2: float = 2.0                         # disjoint gene coefficient
    c3: float = 0.3                         # weight difference coefficient (baseline ~0.42)
    species_stagnation_limit: int = 60      # generations before species culled

    # ── Species interactions ──────────────────────────────────────────────────
    interaction_decay_rate: float = 0.97    # multiplied every 100 ticks (0.97^100 ≈ 0.05)
    interaction_decay_interval: int = 100   # ticks between decay passes
    ticks_per_year: int = 5000              # for age-in-years display

    # ── Reproduction ──────────────────────────────────────────────────────────
    mate_search_radius: float = 80.0

    # ── Naming ────────────────────────────────────────────────────────────────
    name_min_syllables: int = 2
    name_max_syllables: int = 4
    syllable_mutation_rate: float = 0.15    # probability a syllable mutates

    # ── Lineage ───────────────────────────────────────────────────────────────
    max_lineage_generations: int = 100

    # ── Spatial Hash ──────────────────────────────────────────────────────────
    spatial_cell_size: float = 150.0

    # ── Visualization ─────────────────────────────────────────────────────────
    window_width: int = 1280
    window_height: int = 720
    target_fps: int = 60
    show_trails: bool = True
    trail_length: int = 20
    show_rays: bool = True
    show_signals: bool = False       # draw broadcast-signal aura around creatures
    show_sense_radius: bool = False

    # ── Camera ────────────────────────────────────────────────────────────────
    camera_pan_speed: float = 8.0
    camera_zoom_speed: float = 0.1
    camera_min_zoom: float = 0.15
    camera_max_zoom: float = 4.0
    camera_initial_zoom: float = 0.25

    # ── Save / Load ───────────────────────────────────────────────────────────
    save_dir: str = "saves"
    autosave_enabled: bool = True
    autosave_interval_ticks: int = 3000

    # ── Simulation speed ──────────────────────────────────────────────────────
    sim_speed: int = 1                      # ticks per frame; [ / ] keys cycle presets

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats_window: int = 300                 # rolling average window in ticks


# Singleton — import this everywhere
CONFIG = SimConfig()
