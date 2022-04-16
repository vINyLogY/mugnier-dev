# coding: utf-8
"""Metas."""
from __future__ import annotations
from weakref import WeakValueDictionary
from builtins import map, zip
from itertools import tee
from operator import itemgetter
from typing import Any, Callable, Generator, Iterable, Literal, Optional, TypeVar


class Cached(type):
    """Meta-class to create a class that auto generate cached class.

    Args
    ----
    identifier : str
        Parameter used in __init__ method in class to identify a object.
    default : a
        The value of the parameter that will not be used as an identifier in
        the cache. 
    """

    def __new__(cls, name: str, bases: tuple, dct: dict, **kwarg) -> Cached:
        return super().__new__(cls, name, bases, dct)

    def __init__(self,
                 name: str,
                 bases: tuple,
                 dct: dict,
                 identifier: str = 'name',
                 default: Any = None) -> None:
        self.__identifier = identifier
        self.__default = default
        self.__cache = WeakValueDictionary()
        super().__init__(name, bases, dct)
        return

    def __call__(self, *args, **kwargs) -> Any:
        identifier = self.__identifier
        default = self.__default
        if identifier in kwargs and kwargs[identifier] != default:
            if str(kwargs[identifier]) in self.__cache:
                obj = self.__cache[str(kwargs[identifier])]
            else:
                obj = super().__call__(*args, **kwargs)
                self.__cache[str(kwargs[identifier])] = obj
        else:
            obj = super().__call__(*args, **kwargs)
        return obj


A = TypeVar('A')
B = TypeVar('B')

def lazyproperty(func: Callable[..., A]) -> Callable[..., A]:
    name = '__lazy_' + func.__name__

    @property
    def lazy(self) -> A:
        if hasattr(self, name):
            return getattr(self, name)
        else:
            value = func(self)
            setattr(self, name, value)
            return value

    return lazy


T = TypeVar('T')


def iter_visitor(start: T,
                 r: Callable[[T], list[T]],
                 method: Literal['DFS' , 'BFS'] = 'DFS') -> Generator[T, None, None]:
    """Iterative visitor.

    Args:
        start: Initial object
        r: Relation function.
        method: in {'DFS', 'BFS'}. 'DFS': Depth first; 'BFS': Breadth first.
    """
    stack, visited = [start], set()
    while stack:
        if method == 'DFS':
            vertex = stack.pop()
        elif method == 'BFS':
            vertex, stack = stack[0], stack[1:]
        else:
            raise NotImplementedError(
                "Only support 'DFS' (depth first) and 'BFS' (breadth first) methods."
            )
        if vertex not in visited:
            visited.add(vertex)
            try:
                n_ = r(vertex)
                stack.extend(n_ - visited)
            except:
                pass
            yield vertex


def huffman_tree(sources: list(T),
                 importance: Optional[list[int]] = None,
                 obj_new: Optional[Callable[[], T]] = None,
                 n_branch: int = 2) -> T:

    def obj(x):
        return x[0]

    def key(x):
        return x[1]

    if importance is None:
        importance = [1] * len(sources)
    if obj_new is None:

        class counter:
            n = 0

            def __new__(cls) -> int:
                cls.n += 1
                return cls.n

        obj_new = counter

    sequence = list(zip(sources, importance))
    graph = {}
    while len(sequence) > 1:
        sequence.sort(key=key)
        try:
            branch, sequence = sequence[:n_branch], sequence[n_branch:]
        except:
            branch, sequence = sequence, []
        p = sum(map(key, branch))
        new = obj_new()
        graph[new] = list(map(obj, branch))
        sequence.insert(0, (new, p))
    return graph, obj(sequence[0])


def unzip(iterable: Iterable) -> Iterable[Iterable]:
    """The same as zip(*iter) but returns iterators, instead
    of expand the iterator. Mostly used for large sequence.
    Reference: https://gist.github.com/andrix/1063340

    Args:
        iterable

    Returns
        unzipped iterators
    """
    _tmp, iterable = tee(iterable, 2)
    iters = tee(iterable, len(next(_tmp)))
    return (map(itemgetter(i), it) for i, it in enumerate(iters))
