"""
main.py — entry point for the evolution simulation.

Controls:
  WASD / Arrow keys  — pan camera
  Scroll wheel       — zoom in/out
  F                  — fit entire world on screen
  Click creature     — follow it + open family tree
  T                  — close family tree
  P / Space          — pause / unpause
  S                  — save
  L                  — load most recent save
  E                  — export genome of followed creature
  R                  — toggle ray visualization
  Trails             — toggle via config.show_trails
  Escape / Q         — quit
"""

import sys
import pygame

from config import CONFIG
from core.creature import Creature, reset_id_counter
from core.genome import make_minimal_genome, INNOVATION_REGISTRY
from core.lineage import LineageRegistry
from core.naming import random_name
from core.population import Population
from core.world import World
from data.serializer import Serializer
from data.stats import StatsCollector
from visualization.event_log import get_log, MILESTONE
from visualization.hud import SPEED_PRESETS
from evolution.speciation import Speciator
from simulation.loop import SimulationLoop, _next_lineage_id
from visualization.renderer import Renderer


def make_initial_population(
    population: Population,
    lineage: LineageRegistry,
    loop: SimulationLoop,
) -> None:
    for _ in range(CONFIG.initial_population):
        genome = make_minimal_genome(
            CONFIG.neat_input_size,
            CONFIG.neat_output_size,
            INNOVATION_REGISTRY,
            CONFIG.weight_init_range,
        )
        genome.lineage_id = _next_lineage_id()
        name = random_name(CONFIG.name_min_syllables, CONFIG.name_max_syllables)
        pos = population.random_spawn_position()
        creature = Creature(genome, name, pos, CONFIG.initial_energy)
        population.add(creature)
        lineage.register_birth(creature, 0)


def main() -> None:
    pygame.init()
    pygame.font.init()

    # ── Build simulation objects ───────────────────────────────────────────────
    world      = World(CONFIG)
    population = Population(world, CONFIG)
    lineage    = LineageRegistry(CONFIG.max_lineage_generations)
    loop       = SimulationLoop(world, population, lineage, CONFIG)
    renderer   = Renderer(CONFIG)
    serializer = Serializer(CONFIG)
    stats      = StatsCollector(CONFIG.stats_window)

    make_initial_population(population, lineage, loop)

    # Initial speciation
    loop.speciator.speciate(population.creatures)
    get_log().push(MILESTONE, f"Simulation started — {CONFIG.initial_population} creatures", 0)

    paused = False
    pan_keys = {
        pygame.K_LEFT: (-1, 0), pygame.K_RIGHT: (1, 0),
        pygame.K_UP:   (0, -1), pygame.K_DOWN:  (0, 1),
        pygame.K_a:    (-1, 0), pygame.K_d:     (1, 0),
        pygame.K_w:    (0, -1), pygame.K_s:     (0, 1),
    }

    running = True
    while running:
        sw, sh = renderer.screen.get_size()

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Options menu gets first crack
            if renderer.options.handle_event(event, sw, sh):
                continue

            # Creature picker second
            if renderer.picker.handle_event(event):
                continue

            # Detail panel third
            panel_x = renderer.panel_x()
            if renderer.detail.handle_event(event, panel_x):
                continue

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    if renderer.detail.visible:
                        renderer.detail.close()
                        renderer.camera.follow(-1)
                    else:
                        running = False

                elif event.key in (pygame.K_p, pygame.K_SPACE):
                    paused = not paused

                elif event.key == pygame.K_f:
                    renderer.camera.snap_to_world(sw, sh)

                elif event.key == pygame.K_t:
                    renderer.detail.close()
                    renderer.camera.follow(-1)

                elif event.key == pygame.K_o:
                    renderer.options.toggle()

                elif event.key == pygame.K_r:
                    CONFIG.show_rays = not CONFIG.show_rays

                elif event.key == pygame.K_g:
                    CONFIG.show_signals = not CONFIG.show_signals

                elif event.key == pygame.K_RIGHTBRACKET:
                    idx = SPEED_PRESETS.index(CONFIG.sim_speed) if CONFIG.sim_speed in SPEED_PRESETS else 0
                    CONFIG.sim_speed = SPEED_PRESETS[min(idx + 1, len(SPEED_PRESETS) - 1)]

                elif event.key == pygame.K_LEFTBRACKET:
                    idx = SPEED_PRESETS.index(CONFIG.sim_speed) if CONFIG.sim_speed in SPEED_PRESETS else 0
                    CONFIG.sim_speed = SPEED_PRESETS[max(idx - 1, 0)]

                elif event.key == pygame.K_s:
                    serializer.save(loop, population, lineage, loop.speciator)

                elif event.key == pygame.K_l:
                    saves = serializer.list_saves()
                    if saves:
                        serializer.load(loop, population, lineage, loop.speciator, saves[0]["name"])
                    else:
                        print("[Load] No saves found.")

                elif event.key == pygame.K_i:
                    entries = serializer.list_saved_creatures()
                    renderer.picker.toggle(entries)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx = event.pos[0]
                    my = event.pos[1]

                    # ── HUD click zones (species list / oldest alive) ──────────
                    hud_cid = renderer.hud.get_creature_at(mx, my)
                    if hud_cid is not None:
                        c = population.get_by_id(hud_cid)
                        if c:
                            renderer.camera.follow(hud_cid)
                            renderer.detail.open(c, lineage)
                        continue  # don't fall through to world pick

                    # ── World area click ──────────────────────────────────────
                    world_right = panel_x if renderer.detail.visible else sw
                    if mx < world_right:
                        cid = renderer.pick_creature(event.pos[0], event.pos[1], population)
                        if cid is not None:
                            c = population.get_by_id(cid)
                            if c:
                                renderer.camera.follow(cid)
                                renderer.detail.open(c, lineage)
                        else:
                            renderer.detail.close()
                            renderer.camera.follow(-1)

                elif event.button == 4:   # scroll up
                    if not renderer.detail.visible and event.pos[0] > sw - 270:
                        get_log().handle_scroll(+1)
                    elif not renderer.detail.visible:
                        renderer.camera.zoom_around(event.pos[0], event.pos[1], 1.12)
                elif event.button == 5:   # scroll down
                    if not renderer.detail.visible and event.pos[0] > sw - 270:
                        get_log().handle_scroll(-1)
                    elif not renderer.detail.visible:
                        renderer.camera.zoom_around(event.pos[0], event.pos[1], 1 / 1.12)

            elif event.type == pygame.MOUSEWHEEL:
                if not renderer.detail.visible:
                    mx, my = pygame.mouse.get_pos()
                    renderer.camera.zoom_around(mx, my, 1.12 if event.y > 0 else 1 / 1.12)

            elif event.type == pygame.VIDEORESIZE:
                pass  # RESIZABLE flag handles this automatically

        # ── Camera pan ────────────────────────────────────────────────────────
        if not renderer.detail.visible and not renderer.options.visible:
            keys = pygame.key.get_pressed()
            pan_speed = CONFIG.camera_pan_speed
            for key, (dx, dy) in pan_keys.items():
                if keys[key]:
                    renderer.camera.pan(dx * pan_speed, dy * pan_speed)

        # ── Save creature request (from detail panel Save button) ─────────────
        if renderer.detail.save_requested:
            renderer.detail.save_requested = False
            c = population.get_by_id(renderer.detail.creature_id)
            if c:
                serializer.save_creature(c, loop.tick_count)

        # ── Spawn copy request (from detail panel Spawn Copy button) ──────────
        if renderer.detail.spawn_requested:
            renderer.detail.spawn_requested = False
            src = population.get_by_id(renderer.detail.creature_id)
            if src and population.count < CONFIG.max_population:
                from core.genome import INNOVATION_REGISTRY
                from evolution.mutation import mutate
                child_genome = src.genome.copy()
                mutate(child_genome, CONFIG, INNOVATION_REGISTRY)
                from core.naming import inherit_name
                child_name = inherit_name(src.name, src.name,
                                          CONFIG.syllable_mutation_rate,
                                          min_syllables=CONFIG.name_min_syllables,
                                          max_syllables=CONFIG.name_max_syllables)
                child_pos  = population.random_spawn_position()
                from core.creature import Creature
                child = Creature(child_genome, child_name, child_pos, CONFIG.initial_energy)
                child.parent_ids = (src.id, src.id)
                population.add(child)
                lineage.register_birth(child, loop.tick_count)
                loop._total_births += 1
                from visualization.event_log import push as log_event, BIRTH
                log_event(BIRTH,
                          f"Spawned copy of {src.name.short()} → {child.name.short()}",
                          loop.tick_count)

        # ── Load creature request (from picker modal) ─────────────────────────
        if renderer.picker.load_requested:
            filename = renderer.picker.load_requested
            renderer.picker.load_requested = None
            creature = serializer.load_creature(filename)
            if creature:
                creature.pos = population.random_spawn_position()
                population.add(creature)
                lineage.register_birth(creature, loop.tick_count)
                from visualization.event_log import push as log_event, INJECT
                log_event(INJECT, f"Loaded {creature.name.short()} from file", loop.tick_count)

        # ── Simulation tick(s) ────────────────────────────────────────────────
        if not paused:
            for _ in range(CONFIG.sim_speed):
                loop.tick()
                stats.record(population, loop)

                # Autosave
                if (CONFIG.autosave_enabled
                        and loop.tick_count > 0
                        and loop.tick_count % CONFIG.autosave_interval_ticks == 0):
                    serializer.save(loop, population, lineage, loop.speciator, name="autosave")

        # ── Render ────────────────────────────────────────────────────────────
        renderer.render(loop, population, lineage, loop.speciator, paused)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
