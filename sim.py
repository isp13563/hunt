#!/usr/bin/env python3
"""Predator/prey simulation."""
import argparse
import enum
import math
import random
from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy
from matplotlib.animation import FuncAnimation


class AnimalType(enum.IntEnum):
    NONE = 0
    PREY = 1
    PREDATOR = 2


@dataclass(eq=True, frozen=True)
class Position:
    """Base position class."""

    x: int
    y: int


@dataclass
class Animal(ABC):
    """Base animal class."""

    pos: Position
    sight: int
    type_ = AnimalType.NONE
    nearest_animals: List["Animal"] = field(default_factory=list)

    @property
    def x(self) -> int:
        return self.pos.x

    @property
    def y(self) -> int:
        return self.pos.y

    def update(self, all_animals: Iterable["Animal"]) -> None:
        """Update knowledge about nearest animals."""
        self.nearest_animals = sorted(
            (
                animal
                for animal in all_animals
                if animal != self and distance(self, animal) <= self.sight
            ),
            key=lambda animal: distance(self, animal),
        )


def distance(animal1: Animal, animal2: Animal) -> float:
    """Realistic distance between positions."""
    return round(
        math.sqrt((animal2.x - animal1.x) ** 2 + (animal2.y - animal1.y) ** 2),
        2,
    )


@dataclass
class Prey(Animal):
    """Prey class."""

    sight: int
    type_ = AnimalType.PREY


@dataclass
class Predator(Animal):
    """Predator class."""

    sight: int
    type_ = AnimalType.PREDATOR


def get_closest_prey(animal: Animal) -> Optional[Animal]:
    """Get closest prey in sight."""
    closest_preys = [a for a in animal.nearest_animals if isinstance(a, Prey)]
    if closest_preys:
        return closest_preys[0]
    return None


def get_closest_predator(animal: Animal) -> Optional[Animal]:
    """Get closest predator in sight."""
    closest_predators = [a for a in animal.nearest_animals if isinstance(a, Predator)]
    if closest_predators:
        return closest_predators[0]
    return None


@dataclass
class World:
    """World class that uses numpy array to store animals."""

    side: int
    prey_sight: int
    predator_sight: int
    grid: numpy.ndarray = field(init=False)
    animals: Dict[Position, Animal] = field(default_factory=dict, init=False)

    def get_preys(self) -> List[Prey]:
        return [a for a in self.animals if isinstance(a, Prey)]

    def __post_init__(self) -> None:
        self.grid = numpy.zeros((self.side, self.side))

    def is_free(self, x, y) -> bool:
        return self.grid[x][y] == AnimalType.NONE.value

    def spawn(self, animal: Animal, update: bool = True):
        """Add an animal to the grid."""
        self.grid[animal.x][animal.y] = animal.type_.value
        self.animals[animal.pos] = animal
        if update:
            self.update()

    def spawn_random(self, type_: AnimalType, update: bool = True) -> None:
        if type_ == AnimalType.NONE:
            raise RuntimeError("Can random spawn either prey or predator")
        available_slots = self.side * self.side - len(self.animals)
        animal: Animal
        while available_slots:
            rand_x = random.randint(0, self.side - 1)
            rand_y = random.randint(0, self.side - 1)
            if not self.is_free(rand_x, rand_y):
                continue
            pos = Position(rand_x, rand_y)
            self.spawn(
                Prey(pos, self.prey_sight)
                if type_ == AnimalType.PREY
                else Predator(pos, self.predator_sight)
            )
            if update:
                self.update()
            return None

    def spawn_prey(
        self, location: Optional[Tuple[int, int]] = None, update: bool = True
    ):
        if location is not None:
            self.spawn(Prey(Position(*location), self.prey_sight), update=update)
        else:
            self.spawn_random(AnimalType.PREY, update=update)

    def spawn_predator(
        self, location: Optional[Tuple[int, int]] = None, update: bool = True
    ):
        if location is not None:
            self.spawn(
                Predator(Position(*location), self.predator_sight), update=update
            )
        else:
            self.spawn_random(AnimalType.PREDATOR, update=update)

    def move(
        self, animal: Animal, new_position: Position, update: bool = False
    ) -> None:
        if not self.is_free(new_position.x, new_position.y):
            return None
        self.animals.pop(animal.pos, None)
        self.grid[animal.x][animal.y] = AnimalType.NONE.value
        self.grid[new_position.x][new_position.y] = animal.type_.value
        animal.pos = Position(new_position.x, new_position.y)
        self.animals[animal.pos] = animal
        if update:
            self.update()

    def kill(self, animal: Animal, update: bool = False) -> None:
        """Make an animal disappear from the grid."""
        if self.is_free(animal.x, animal.y) or isinstance(animal, Predator):
            return None
        self.animals.pop(animal.pos, None)
        self.grid[animal.x][animal.y] = 0
        if update:
            self.update()

    def update(self) -> None:
        """Update animal knowledge."""
        for animal in self.animals.values():
            animal.update(self.animals.values())

    def run_once(self):
        for animal in list(self.animals.values()):
            if animal.type_ == AnimalType.PREY:
                move_prey(animal, self)
        for animal in list(self.animals.values()):
            if animal.type_ == AnimalType.PREDATOR:
                move_predator(animal, self)
        self.update()


def move_random(animal: Animal, world: World) -> Position:
    """Move to a random direction."""
    print(f"{animal.type_.name} at {(animal.x, animal.y)} moves randomly")
    all_directions = (
        (animal.x - 1, animal.y),
        (animal.x + 1, animal.y),
        (animal.x, animal.y - 1),
        (animal.x, animal.y + 1),
        (animal.x, animal.y),  # do not move
    )
    possible_directions = [
        (x, y)
        for (x, y) in all_directions
        if in_grid(x, y, world) and world.is_free(x, y)
    ]
    if not possible_directions:
        return Position(animal.x, animal.y)
    else:
        return Position(*random.choice(possible_directions))


def in_grid(x: int, y: int, world: World) -> bool:
    return 0 <= x < world.side and 0 <= y < world.side


def closest_to_border(x: int, y: int, world: World) -> Tuple[int, int]:
    """Find closest to border point."""
    new_x = x
    new_y = y
    if x >= world.side:
        new_x = world.side - 1
    elif x < 0:
        new_x = 0

    if y >= world.side:
        new_y = world.side - 1
    elif y < 0:
        new_y = 0
    return (new_x, new_y)


def move_closer(animal: Animal, target: Animal, world: World) -> Position:
    """Move closer to target."""
    print(f"{animal.type_.name} chases {target.type_.name}")
    dy = target.y - animal.y
    dx = target.x - animal.x
    new_y = animal.y
    new_x = animal.x
    if abs(dy) >= abs(dx):
        if dy > 0:
            new_y += 1
        elif dy < 0:
            new_y -= 1
    else:
        if dx > 0:
            new_x += 1
        elif dx < 0:
            new_x -= 1

    new_x, new_y = closest_to_border(new_x, new_y, world)
    return Position(new_x, new_y)


def move_further(animal: Animal, target: Animal, world: World) -> Position:
    """Move further from target."""
    print(f"{animal.type_.name} runs from {target.type_.name}")
    dy = target.y - animal.y
    dx = target.x - animal.x
    new_y = animal.y
    new_x = animal.x
    if abs(dy) >= abs(dx):
        if dy > 0:
            new_y -= 1
        elif dy < 0:
            new_y += 1
    else:
        if dx > 0:
            new_x -= 1
        elif dx < 0:
            new_x += 1

    new_x, new_y = closest_to_border(new_x, new_y, world)
    return Position(new_x, new_y)


def move_prey(prey: Prey, world: World) -> None:
    """Prey strategy."""
    closest_predator = get_closest_predator(prey)
    closest_prey = get_closest_prey(prey)

    new_position: Position
    if closest_predator is None and closest_prey is None:
        new_position = move_random(prey, world)
    elif closest_predator is not None:
        new_position = move_further(prey, closest_predator, world)
    else:
        assert closest_prey is not None  # make mypy shut up
        # add some randomness to preys so that they don't stay glued
        if distance(prey, closest_prey) == 1:
            new_position = move_random(prey, world)
        else:
            new_position = move_closer(prey, closest_prey, world)
    world.move(prey, new_position)


def move_predator(predator: Predator, world: World) -> None:
    """Predator strategy."""
    closest_prey = get_closest_prey(predator)
    new_position: Position
    if closest_prey is None:
        new_position = move_random(predator, world)
        world.move(predator, new_position)
    elif distance(predator, closest_prey) > 1:
        new_position = move_closer(predator, closest_prey, world)
        world.move(predator, new_position)
    else:  # kill adjacent prey
        world.kill(closest_prey)
        return None


def test_preys(side=50, prey_sight=10, predator_sight=30):
    world = World(side=side, prey_sight=prey_sight, predator_sight=predator_sight)
    assert world.animals == {}

    # Add first animal
    first_prey = Prey(Position(0, 0), prey_sight)
    world.spawn(first_prey)
    assert get_closest_prey(first_prey) is None

    # Add second animal within sight
    another_prey = Prey(Position(2, 2), prey_sight)
    world.spawn(another_prey)
    assert get_closest_prey(first_prey) == another_prey
    assert get_closest_prey(another_prey) == first_prey

    # Add prey far away
    far_away_prey = Prey(Position(world.side - 1, world.side - 1), prey_sight)
    world.spawn(far_away_prey)
    assert get_closest_prey(first_prey) == another_prey
    assert get_closest_prey(another_prey) == first_prey
    assert get_closest_prey(far_away_prey) is None

    # Kill preys
    world.kill(far_away_prey)
    world.kill(another_prey)
    world.kill(first_prey)
    world.update()
    assert world.animals == {}


@dataclass
class Config:
    num_preys: int
    num_predators: int
    square_side: int
    prey_sight: int
    predator_sight: int
    max_iterations: int


def generate_random_world(config: Config) -> World:
    world = World(
        side=config.square_side,
        prey_sight=config.prey_sight,
        predator_sight=config.predator_sight,
    )
    for _ in range(config.num_preys):
        world.spawn_prey()
    for _ in range(config.num_predators):
        world.spawn_predator()
    return world


def animate_world(
    world: World, max_iterations: int = 3, filepath: Optional[str] = None
):
    if max_iterations < 1:
        print("max_iterations must be >= 1")
        return
    fig = plt.figure(figsize=(world.side, world.side))
    image = plt.imshow(world.grid)

    def update(_):
        world.run_once()
        image.set_array(world.grid)

    frames = max_iterations
    animation = FuncAnimation(fig, update, frames=frames, interval=50, repeat=False)
    if filepath:
        animation.save(filepath)
    else:
        plt.show()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preys", type=int, default=70, help="Number of preys")
    parser.add_argument("--predators", type=int, default=10, help="Number of predators")
    parser.add_argument("--prey-sight", type=int, default=10, help="Prey sight")
    parser.add_argument("--predator-sight", type=int, default=30, help="Predator sight")
    parser.add_argument(
        "--grid-size", type=int, default=50, help="Size of the square grid"
    )
    parser.add_argument(
        "--iterations", type=int, default=300, help="Number of simulation iterations"
    )
    args = parser.parse_args()
    config = Config(
        num_preys=args.preys,
        num_predators=args.predators,
        square_side=args.grid_size,
        prey_sight=args.prey_sight,
        predator_sight=args.predator_sight,
        max_iterations=args.iterations,
    )
    world = generate_random_world(config)
    animate_world(world, config.max_iterations)


if __name__ == "__main__":
    test_preys()
    run()

# animate_world(world, max_iterations=10, filepath="/tmp/animation.gif")
