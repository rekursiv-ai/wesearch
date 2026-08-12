"""Declarative parameter specs: one description of a tunable knob.

A knob reached three surfaces by being spelled three times -- a JSON-Schema
property for the sagent tool, a validation branch in that tool's ``run``, and a
signature default on the MCP server -- and the copies drifted every time. The
MCP side kept a ``browser`` bool after sagent grew a five-valued ``transport``;
both descriptions omitted a category the type had always accepted; the default
extractor needed four edits to change.

A :class:`Param` is that description, and a :class:`ParamSet` is a class whose
declared params ARE its fields. Each surface RENDERS the set rather than
restating it: :meth:`ParamSet.json_schema` builds the tool schema,
:meth:`ParamSet.coerce` validates a directive against the same types, and
:meth:`ParamSet.default` supplies a signature default.

No module-level tables and no framework types: a spec is data, so it lives in
``wesearch`` (which exports standalone and cannot import a tool package) while
both adapters read it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, get_args, get_origin

import types
import typing


# Both origins ``get_origin`` can report for a union. On the 3.12 floor
# ``.export/pyproject.toml`` publishes, it returns ``typing.Union`` for
# ``Optional[X]`` and ``types.UnionType`` for ``X | None``; 3.14 unified the two
# (gh-105499). Matching one spelling silently returned () for the other,
# dropping the enum of every optional param while passing on the dev
# interpreter.
#
# ``typing.Union`` is read out of the module namespace rather than written as
# ``typing.Union``: both type checkers and ruff read that name as the
# DEPRECATED annotation form, even here where it is only ever compared as a
# runtime value.
_UNION_ORIGINS: frozenset[object] = frozenset({types.UnionType, vars(typing)["Union"]})


__all__ = ["Param", "ParamSet", "literal_values"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Param[T]:
    """One tunable knob: its type, its default, and its prose.

    Attributes:
      annotation: The parameter's type. A ``Literal`` (or an alias of one)
        yields the schema ``enum`` and the accepted values; any other type is
        checked by ``isinstance``.
      default: Value used when a directive omits the key. ``None`` with
        ``required=True`` means the caller must supply it.
      description: Agent-facing prose, rendered into every surface's
        description so the explanation cannot differ between them.
      required: Whether omission is an error rather than a defaulted value.
      schema_extra: Extra JSON-Schema keys merged into this param's property,
        for a shape the type alone does not carry (a free-form object's
        ``additionalProperties``).

    """

    annotation: object
    default: T | None = None
    description: str = ""
    required: bool = False
    schema_extra: Mapping[str, object] | None = None

    @property
    def choices(self) -> tuple[object, ...]:
        """The ``Literal`` members this param accepts, or ``()`` if unbounded.

        Descends through an optional (``X | None``) and a PEP-695 ``type``
        alias, both of which hide their members from a bare ``get_args``: the
        first yields the inner alias and ``NoneType``, the second yields
        nothing at all.
        """
        return literal_values(self.annotation)

    @property
    def json_type(self) -> str | None:
        """The JSON-Schema ``type`` keyword, or ``None`` when unconstrained.

        ``object`` as the annotation means "any JSON value" -- a POST body is
        as legitimately a list or a string as a mapping -- so the property
        omits ``type`` rather than narrowing it to one shape.
        """
        if self.choices:
            return "string"
        if self.annotation is object:
            return None
        checkable = get_origin(self.annotation) or self.annotation
        return _JSON_TYPES.get(checkable, "object")

    def schema(self) -> dict[str, object]:
        """Render this param as one JSON-Schema property."""
        prop: dict[str, object] = {}
        if self.json_type is not None:
            prop["type"] = self.json_type
        if self.choices:
            prop["enum"] = list(self.choices)
        if self.description:
            prop["description"] = self.description
        if self.schema_extra:
            prop.update(self.schema_extra)
        return prop

    def coerce(self, name: str, value: object) -> object:
        """Return ``value``, having checked it against this param's type.

        Returns ``object``, not ``T``: the check is a RUNTIME one over a JSON
        directive, and claiming a static narrowing the annotation cannot prove
        would push a cast into every caller anyway. Callers that need the
        narrow type cast once, at the boundary, where the ``Literal`` is in
        scope.

        Raises:
          ValueError: When ``value`` is outside the accepted set. Carries the
            valid values, since the caller is usually a model that emitted a
            near-miss.

        """
        if self.choices:
            if value not in self.choices:
                valid = ", ".join(str(c) for c in self.choices)
                raise ValueError(f"Invalid {name} {value!r}. Valid: {valid}.")
            return value
        # The ORIGIN of a subscripted generic: ``isinstance(v, dict[str, str])``
        # raises, so testing ``isinstance(annotation, type)`` alone let every
        # parameterized annotation skip validation entirely.
        checkable = get_origin(self.annotation) or self.annotation
        if isinstance(checkable, type) and not _is_instance(value, checkable):
            raise ValueError(
                f"Invalid {name}: expected {checkable.__name__},"
                f" got {type(value).__name__}."
            )
        return value


class ParamSet:
    """A tool's parameter surface: subclass and declare :class:`Param` fields.

    The declared attribute name IS the parameter name, so there is no second
    place to spell it. Inherited params come first, which lets a surface that
    accepts more (a POST-capable fetch) extend one that accepts less.
    """

    @classmethod
    def params(cls) -> Mapping[str, Param[object]]:
        """Every declared param, base classes first, in declaration order."""
        return {
            name: value
            for klass in reversed(cls.__mro__)
            for name, value in vars(klass).items()
            if isinstance(value, Param)
        }

    @classmethod
    def json_schema(cls) -> dict[str, object]:
        """Render the whole set as a JSON-Schema object."""
        params = cls.params()
        return {
            "type": "object",
            "properties": {name: p.schema() for name, p in params.items()},
            "required": [name for name, p in params.items() if p.required],
        }

    @classmethod
    def coerce(cls, args: Mapping[str, object]) -> dict[str, object]:
        """Return every declared param, narrowed or defaulted.

        Keys the set does not declare are ignored, not rejected: a tool may
        carry its own extras (a POST body) that no spec describes.

        Raises:
          ValueError: On a missing required param or an out-of-range value.

        """
        out: dict[str, object] = {}
        for name, param in cls.params().items():
            if name not in args or args[name] is None:
                if param.required:
                    raise ValueError(f"Missing required parameter {name!r}.")
                out[name] = param.default
                continue
            out[name] = param.coerce(name, args[name])
        return out

    @classmethod
    def asset_markdown(cls) -> str:
        """Render the set as prose bullets for a tool-description asset."""
        lines: list[str] = []
        for name, param in cls.params().items():
            suffix = f" Default `{param.default}`." if param.default is not None else ""
            lines.append(f"- `{name}` -- {param.description}{suffix}")
        return "\n".join(lines)


_JSON_TYPES: Mapping[object, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    tuple: "array",
    dict: "object",
}


def _is_instance(value: object, kind: type) -> bool:
    """``isinstance``, minus Python's bool-is-an-int inheritance.

    ``isinstance(True, int)`` is true, so a directive of ``{"count": true}``
    satisfied an ``int`` param and reached the tool as a boolean. JSON has
    distinct ``true`` and ``1`` literals, and the caller here is a model
    emitting JSON, so the two must not be interchangeable.
    """
    if kind is not bool and isinstance(value, bool):
        return False
    return isinstance(value, kind)


def literal_values(annotation: object) -> tuple[object, ...]:
    """The ``Literal`` members of ``annotation``, or ``()`` for anything else.

    Unwraps a PEP-695 ``type`` alias (whose members hide behind ``__value__``)
    and an optional union, then flattens nested literals.

    The ``Literal`` ORIGIN is required, not merely a non-empty ``get_args``:
    every parameterized generic has arguments, so accepting them turned
    ``dict[str, str]`` into two "choices" -- emitting ``{"type": "string",
    "enum": [<class 'str'>, <class 'str'>]}``, which is not valid JSON Schema,
    and rejecting a real value with "Valid: <class 'str'>".
    """
    annotation = getattr(annotation, "__value__", annotation)
    origin = get_origin(annotation)
    # BOTH union spellings: Python 3.14 unified them (gh-105499), but on 3.12 --
    # the floor ``.export/pyproject.toml`` publishes -- ``get_origin`` returns
    # ``typing.Union`` for ``Optional[X]`` AND for ``X | None``. Matching only
    # ``types.UnionType`` therefore returned () there, silently dropping the
    # enum of every optional param while passing on the dev interpreter.
    if origin in _UNION_ORIGINS:
        values: list[object] = []
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            members = literal_values(arg)
            # EVERY non-None member must be a Literal, or the union as a whole
            # has no fixed choice set. Falling back to the member itself
            # promoted a plain type into an enum, so ``str | None`` emitted
            # ``{"enum": [<class 'str'>]}`` and then rejected a real string
            # with "Valid: <class 'str'>".
            if not members:
                return ()
            values.extend(members)
        return tuple(values)
    return get_args(annotation) if origin is Literal else ()
