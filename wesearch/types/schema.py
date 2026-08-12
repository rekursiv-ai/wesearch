"""Declarative descriptions of a callable surface's parameters.

A parameter reached three surfaces by being spelled three times -- a
JSON-Schema property for the sagent tool, a validation branch in that tool's
``run``, and a signature default on the MCP server -- and the copies drifted
every time. The MCP side kept a ``browser`` bool after sagent grew a
five-valued ``transport``; both descriptions omitted a category the type had
always accepted; the default extractor needed four edits to change.

A :class:`Field` is one such description and a :class:`Schema` is a class whose
declared fields ARE its parameters. Each surface RENDERS the schema rather than
restating it: :meth:`Schema.json_schema` builds the tool schema,
:meth:`Schema.coerce` validates a directive against the same types, and a
field's ``default`` supplies a signature default.

Descriptions, not values -- ``types.params`` holds the values a caller passes
to :func:`~wesearch.fetch.fetch`, and this holds the metadata a renderer
reads about them. Nothing here is wesearch-specific, and it imports no
framework type: a schema is data, so it lives in ``wesearch`` (which exports
standalone and cannot import a tool package) while every adapter reads it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, get_args, get_origin

import types
import typing


__all__ = ["Field", "Schema", "literal_values"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Field[T]:
    """One declared parameter: its type, its default, and its prose.

    Attributes:
      annotation: The parameter's type. A ``Literal`` (or an alias of one)
        yields the schema ``enum`` and the accepted values; any other type is
        checked by ``isinstance``.
      default: Value used when a directive omits the key. ``None`` with
        ``required=True`` means the caller must supply it.
      description: Agent-facing prose, rendered into every surface's
        description so the explanation cannot differ between them.
      required: Whether omission is an error rather than a defaulted value.
      schema_extra: Extra JSON-Schema keys merged into this field's property,
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
        """The ``Literal`` members this field accepts, or ``()`` if unbounded.

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
        shape = _JSON_SHAPES.get(self._checkable)
        # An UNKNOWN annotation gets no keyword rather than a guessed one: a
        # union of non-literals has no single JSON shape, and advertising
        # "object" there would promise a mapping the check does not enforce.
        return shape.json_type if shape is not None else None

    @property
    def accepted_types(self) -> tuple[type, ...]:
        """The Python types a decoded JSON value may have, or ``()`` for any.

        Read from the SAME table that produces ``json_type``, so the schema a
        model is shown and the check its directive faces cannot disagree. They
        did: ``float`` advertised ``number`` (which admits ``1``) and then
        rejected ``1`` for not being a ``float``, and ``tuple`` advertised
        ``array`` and rejected the ``list`` every JSON decoder produces.
        """
        shape = _JSON_SHAPES.get(self._checkable)
        return shape.accepts if shape is not None else ()

    @property
    def _checkable(self) -> object:
        """The annotation reduced to something ``isinstance`` can take.

        A subscripted generic reduces to its ORIGIN (``isinstance(v,
        dict[str, str])`` raises); a union of non-literals reduces to nothing,
        since no single type describes it.
        """
        if self.annotation is object:
            return object
        return get_origin(self.annotation) or self.annotation

    def schema(self) -> dict[str, object]:
        """Render this field as one JSON-Schema property."""
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
        """Return ``value``, having checked it against this field's type.

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
        accepted = self.accepted_types
        if accepted and not _is_instance(value, accepted):
            expected = " or ".join(sorted(k.__name__ for k in accepted))
            raise ValueError(
                f"Invalid {name}: expected {expected}, got {type(value).__name__}."
            )
        return value


class Schema:
    """One surface's parameters: subclass and declare :class:`Field` values.

    The declared attribute name IS the parameter name, so there is no second
    place to spell it. Inherited fields come first, which lets a surface that
    accepts more (a POST-capable fetch) extend one that accepts less.
    """

    @classmethod
    def fields(cls) -> Mapping[str, Field[object]]:
        """Every declared field, base classes first, in declaration order."""
        return {
            name: value
            for klass in reversed(cls.__mro__)
            for name, value in vars(klass).items()
            if isinstance(value, Field)
        }

    @classmethod
    def json_schema(cls) -> dict[str, object]:
        """Render the whole set as a JSON-Schema object."""
        fields_ = cls.fields()
        return {
            "type": "object",
            "properties": {name: f.schema() for name, f in fields_.items()},
            "required": [name for name, f in fields_.items() if f.required],
        }

    @classmethod
    def coerce(cls, args: Mapping[str, object]) -> dict[str, object]:
        """Return every declared field, narrowed or defaulted.

        Keys the schema does not declare are ignored, not rejected: a tool may
        carry its own extras (a POST body) that no schema describes.

        Raises:
          ValueError: On a missing required field or an out-of-range value.

        """
        out: dict[str, object] = {}
        for name, field_ in cls.fields().items():
            if name not in args or args[name] is None:
                if field_.required:
                    raise ValueError(f"Missing required parameter {name!r}.")
                out[name] = field_.default
                continue
            out[name] = field_.coerce(name, args[name])
        return out

    @classmethod
    def asset_markdown(cls) -> str:
        """Render the set as prose bullets for a tool-description asset."""
        lines: list[str] = []
        for name, field_ in cls.fields().items():
            default = field_.default
            suffix = f" Default `{default}`." if default is not None else ""
            lines.append(f"- `{name}` -- {field_.description}{suffix}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True, kw_only=True)
class _JsonShape:
    """One JSON type, and the Python types a decoded value of it may have.

    Paired deliberately: the schema keyword and the runtime check are two
    readings of ONE fact, and when they were separate tables they disagreed --
    ``number`` admits a JSON integer but the check demanded ``float``, and
    ``array`` decodes to ``list`` but the check demanded ``tuple``.
    """

    json_type: str | None
    accepts: tuple[type, ...]


_JSON_SHAPES: Mapping[object, _JsonShape] = {
    str: _JsonShape(json_type="string", accepts=(str,)),
    # JSON has one number syntax, so ``1`` is as valid a ``number`` as ``1.0``.
    int: _JsonShape(json_type="integer", accepts=(int,)),
    float: _JsonShape(json_type="number", accepts=(int, float)),
    bool: _JsonShape(json_type="boolean", accepts=(bool,)),
    # Every JSON array decodes to a ``list``; a ``tuple`` annotation describes
    # what the CALLEE wants, not what arrives.
    list: _JsonShape(json_type="array", accepts=(list,)),
    tuple: _JsonShape(json_type="array", accepts=(list, tuple)),
    dict: _JsonShape(json_type="object", accepts=(dict,)),
    # "Any JSON value": no keyword, no check. A union of non-literals lands
    # here too -- no single type describes it, so the schema stays open rather
    # than advertising a shape the check cannot enforce.
    object: _JsonShape(json_type=None, accepts=()),
}


def _is_instance(value: object, kinds: tuple[type, ...]) -> bool:
    """``isinstance`` over several types, minus bool-is-an-int inheritance.

    ``isinstance(True, int)`` is true, so a directive of ``{"count": true}``
    satisfied an ``int`` field and reached the tool as a boolean. JSON has
    distinct ``true`` and ``1`` literals, and the caller here is a model
    emitting JSON, so the two must not be interchangeable.
    """
    if bool not in kinds and isinstance(value, bool):
        return False
    return isinstance(value, kinds)


def literal_values(
    annotation: object,
    *,
    union_origins: frozenset[object] = frozenset(
        {types.UnionType, vars(typing)["Union"]}
    ),
) -> tuple[object, ...]:
    """The ``Literal`` members of ``annotation``, or ``()`` for anything else.

    Unwraps a PEP-695 ``type`` alias (whose members hide behind ``__value__``)
    and an optional union, then flattens nested literals.

    The ``Literal`` ORIGIN is required, not merely a non-empty ``get_args``:
    every parameterized generic has arguments, so accepting them turned
    ``dict[str, str]`` into two "choices" -- emitting ``{"type": "string",
    "enum": [<class 'str'>, <class 'str'>]}``, which is not valid JSON Schema,
    and rejecting a real value with "Valid: <class 'str'>".

    Args:
      annotation: The type to inspect.
      union_origins: Every origin ``get_origin`` may report for a union. Both
        spellings, because 3.14 unified them (gh-105499) while the 3.12 floor
        ``.export/pyproject.toml`` publishes reports ``typing.Union`` for
        ``Optional[X]`` -- matching one silently returned ``()`` for the other,
        dropping the enum of every optional field while passing on the dev
        interpreter. ``typing.Union`` is read out of the module namespace
        because both type checkers and ruff treat the written name as the
        deprecated ANNOTATION form, even compared as a value here.

    Returns:
      members: The literal members, or ``()`` when there is no fixed set.

    """
    annotation = getattr(annotation, "__value__", annotation)
    origin = get_origin(annotation)
    if origin in union_origins:
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
