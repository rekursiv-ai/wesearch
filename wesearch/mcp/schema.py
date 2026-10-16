"""Declarative descriptions of a callable surface's parameters.

A :class:`Field` describes one parameter and a :class:`Schema` is a class whose
declared fields are its parameters. Each surface renders the schema:
:meth:`Schema.json_schema` builds a JSON-Schema tool schema,
:meth:`Schema.coerce` validates a directive against the same types, and
:meth:`Schema.asset_markdown` renders prose bullets.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, cast, get_args, get_origin

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
        description.
      required: Whether omission is an error rather than a defaulted value.
      schema_extra: Extra JSON-Schema keys merged into this field's property,
        for a shape the type alone does not carry (a free-form object's
        ``additionalProperties``).

    """

    annotation: object
    default: T | None = None
    description: str = ""
    required: bool = False
    schema_extra: Mapping[str, object] = field(default_factory=dict[str, object])

    @property
    def choices(self) -> tuple[object, ...]:
        """The ``Literal`` members this field accepts, or ``()`` if unbounded.

        Descends through an optional (``X | None``) and a PEP-695 ``type`` alias.
        """
        return literal_values(self.annotation)

    @property
    def json_type(self) -> str | None:
        """The JSON-Schema ``type`` keyword, or ``None`` when unconstrained.

        An ``object`` annotation (any JSON value) or an annotation with no single
        JSON shape yields ``None``.
        """
        if self.choices:
            return "string"
        shape = self._shape
        return shape.json_type if shape is not None else None

    @property
    def accepted_types(self) -> tuple[type, ...]:
        """The Python types a decoded JSON value may have, or ``()`` for any."""
        shape = self._shape
        return shape.accepts if shape is not None else ()

    @property
    def _shape(self) -> _JsonShape | None:
        """The JSON shape of this field's annotation, or ``None`` if it has none."""
        # A subscripted generic reduces to its origin: ``isinstance(v, dict[str, str])``
        # raises.
        checkable = (
            object
            if self.annotation is object
            else get_origin(self.annotation) or self.annotation
        )
        shapes: Mapping[object, _JsonShape] = {
            str: _JsonShape(json_type="string", accepts=(str,)),
            # JSON has one number syntax, so ``1`` is as valid a ``number`` as ``1.0``.
            int: _JsonShape(json_type="integer", accepts=(int,)),
            float: _JsonShape(json_type="number", accepts=(int, float)),
            bool: _JsonShape(json_type="boolean", accepts=(bool,)),
            # Every JSON array decodes to a ``list``; a ``tuple`` annotation
            # describes what the CALLEE wants, not what arrives.
            list: _JsonShape(json_type="array", accepts=(list,)),
            tuple: _JsonShape(json_type="array", accepts=(list, tuple)),
            dict: _JsonShape(json_type="object", accepts=(dict,)),
            # "Any JSON value": no keyword, no check.
            object: _JsonShape(json_type=None, accepts=()),
        }
        return shapes.get(checkable)

    def schema(self) -> dict[str, object]:
        """Render this field as one JSON-Schema property.

        Returns:
          prop: JSON-Schema fragment (type, enum, description, etc.).

        """
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
        """Return ``value`` unchanged after checking it against this field's type.

        Args:
          name: Field name, used in error messages.
          value: Decoded JSON value to validate.

        Returns:
          value: ``value``, typed ``object``; callers needing ``T`` cast it.

        Raises:
          ValueError: When ``value`` is outside the accepted choices (the message
            lists them) or has a type the field does not accept.

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
                f"Invalid {name}: expected {expected}, got {type(value).__name__}.",
            )
        return value


def literal_values(
    annotation: object,
    *,
    union_origins: frozenset[object] = frozenset(
        {types.UnionType, vars(typing)["Union"]},
    ),
) -> tuple[object, ...]:
    """Return the ``Literal`` members of ``annotation``, or ``()`` for anything else.

    Unwraps a PEP-695 ``type`` alias and an optional union, then flattens nested
    literals. A union with any non-``None``, non-literal member yields ``()``.

    Args:
      annotation: The type to inspect.
      union_origins: Every origin ``get_origin`` may report for a union.
        Python 3.12 reports ``typing.Union`` for ``Optional[X]``; 3.14 reports
        ``types.UnionType``. ``typing.Union`` is read via ``vars`` because the
        type checkers and ruff flag the written name as a deprecated annotation.

    Returns:
      members: The literal members, or ``()`` when there is no fixed set.

    """
    annotation = getattr(annotation, "__value__", annotation)
    origin = get_origin(annotation)
    if origin in union_origins:
        values: list[object] = []
        for arg in cast(tuple[object, ...], get_args(annotation)):
            if arg is type(None):
                continue
            members = literal_values(arg)
            if not members:
                return ()
            values.extend(members)
        return tuple(values)
    return get_args(annotation) if origin is Literal else ()


class Schema:
    """One surface's parameters: subclass and declare :class:`Field` values.

    Each declared attribute name is the parameter name. Inherited fields come
    first, so a subclass extends its base's parameters.
    """

    @classmethod
    def fields(cls) -> Mapping[str, Field[object]]:
        """Every declared field, base classes first, in declaration order.

        Returns:
          fields: Parameter name to its :class:`Field`.

        """
        fields_: dict[str, Field[object]] = {}
        for klass in reversed(cls.__mro__):
            # The stub types ``vars(type)`` as ``MappingProxyType[str, Any]``;
            # widening to ``object`` here makes the isinstance the narrowing.
            namespace = cast(Mapping[str, object], vars(klass))
            fields_ |= {k: v for k, v in namespace.items() if isinstance(v, Field)}
        return fields_

    @classmethod
    def json_schema(cls) -> dict[str, object]:
        """Render every field as one JSON-Schema object.

        Returns:
          schema: JSON-Schema with ``type``, ``properties``, and ``required``.

        """
        fields_ = cls.fields()
        return {
            "type": "object",
            "properties": {name: f.schema() for name, f in fields_.items()},
            "required": [name for name, f in fields_.items() if f.required],
        }

    @classmethod
    def coerce(cls, args: Mapping[str, object]) -> dict[str, object]:
        """Return every declared field, validated or defaulted.

        Undeclared keys in ``args`` are ignored. A ``None`` value counts as
        omitted.

        Args:
          args: Directive arguments, keyed by parameter name.

        Returns:
          out: One entry per declared field, defaults applied.

        Raises:
          ValueError: On a missing required field or an invalid value.

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
        """Render every field as a Markdown bullet for a tool description.

        Returns:
          markdown: One bullet per field: name, description, and default.

        """
        lines: list[str] = []
        for name, field_ in cls.fields().items():
            default = field_.default
            suffix = f" Default `{default}`." if default is not None else ""
            lines.append(f"- `{name}` -- {field_.description}{suffix}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True, kw_only=True)
class _JsonShape:
    """One JSON type, and the Python types a decoded value of it may have.

    One table feeds both the schema keyword and the runtime check, so the two
    cannot disagree.
    """

    json_type: str | None
    accepts: tuple[type, ...]


# ``isinstance(True, int)`` is true, but JSON ``true`` and ``1`` must not be
# interchangeable.
def _is_instance(value: object, kinds: tuple[type, ...]) -> bool:
    """``isinstance`` over several types, minus bool-is-an-int inheritance."""
    if bool not in kinds and isinstance(value, bool):
        return False
    return isinstance(value, kinds)
