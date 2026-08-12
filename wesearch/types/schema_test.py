"""Tests for the declarative parameter descriptions in ``types.schema``."""

from __future__ import annotations

from typing import Literal

import typing

import pytest

from wesearch.types.schema import Field, Schema, literal_values


_Color = Literal["red", "green"]
type _Alias = Literal["a", "b"]


class _Base(Schema):
    name = Field[str](annotation=str, required=True, description="A name.")
    color = Field[_Color](annotation=_Color, default="red", description="A color.")


class _Derived(_Base):
    count = Field[int](annotation=int, default=1, description="How many.")


def test_params_include_inherited_base_first() -> None:
    """A surface that accepts more extends one that accepts less."""
    assert list(_Derived.fields()) == ["name", "color", "count"]


def test_json_schema_renders_types_enums_and_required() -> None:
    schema = _Derived.json_schema()
    assert schema["required"] == ["name"]
    props = schema["properties"]
    assert isinstance(props, dict)
    assert props["color"] == {
        "type": "string",
        "enum": ["red", "green"],
        "description": "A color.",
    }
    assert props["count"] == {"type": "integer", "description": "How many."}


def test_coerce_defaults_the_omitted() -> None:
    assert _Derived.coerce({"name": "x"}) == {"name": "x", "color": "red", "count": 1}


def test_coerce_rejects_a_value_outside_the_literal() -> None:
    with pytest.raises(ValueError, match=r"Invalid color 'blue'\. Valid: red, green\."):
        _Derived.coerce({"name": "x", "color": "blue"})


def test_coerce_rejects_a_wrong_type() -> None:
    with pytest.raises(ValueError, match="expected str, got int"):
        _Derived.coerce({"name": 1})


def test_coerce_requires_a_required_param() -> None:
    with pytest.raises(ValueError, match="Missing required parameter 'name'"):
        _Derived.coerce({})


def test_coerce_ignores_undeclared_keys() -> None:
    """A tool may carry extras no spec describes; they are not rejected here."""
    assert "extra" not in _Derived.coerce({"name": "x", "extra": 1})


def test_literal_values_unwraps_alias_and_optional() -> None:
    """Both hide their members from a bare ``get_args``.

    An optional yields the inner alias plus ``NoneType``; a PEP-695 ``type``
    alias yields nothing at all, so a comparison against it passes vacuously.
    """
    assert literal_values(_Alias) == ("a", "b")
    assert set(literal_values(_Color | None)) == {"red", "green"}
    assert literal_values(str) == ()


def test_coerce_rejects_a_bool_for_an_int() -> None:
    """``isinstance(True, int)`` is true; JSON's ``true`` and ``1`` are not.

    A directive of ``{"count": true}`` satisfied an ``int`` field and reached
    the tool as a boolean.
    """
    with pytest.raises(ValueError, match="expected int, got bool"):
        _Derived.coerce({"name": "x", "count": True})


def test_a_parameterized_generic_is_not_a_literal() -> None:
    """Every generic has ``get_args``; only a ``Literal`` has CHOICES.

    Accepting the former made ``dict[str, str]`` render
    ``{"type": "string", "enum": [<class 'str'>, <class 'str'>]}`` -- not valid
    JSON Schema -- and reject a value with "Valid: <class 'str'>".
    """
    field = Field[dict[str, str]](annotation=dict[str, str])
    assert field.choices == ()
    assert field.schema() == {"type": "object"}
    with pytest.raises(ValueError, match="expected dict, got list"):
        field.coerce("form", [])


def test_both_union_spellings_yield_the_same_members() -> None:
    """``Optional[X]`` and ``X | None`` must agree, on every supported Python.

    3.14 unified the two origins (gh-105499); 3.12 -- the floor the export
    publishes -- reports ``typing.Union`` for BOTH. Matching only
    ``types.UnionType`` returned () there while passing on the dev
    interpreter, silently dropping the enum of every optional field.
    """
    # Built rather than written: ``Optional[X]`` as source trips UP045, and
    # the point is precisely that the deprecated spelling -- which a caller's
    # annotation may still use -- must resolve on the 3.12 floor, where
    # get_origin reports ``typing.Union`` for BOTH spellings.
    optional_color = getattr(typing, "Optional")[_Color]  # noqa: B009 -- see above
    assert literal_values(optional_color) == ("red", "green")
    assert literal_values(_Color | None) == ("red", "green")


def test_an_optional_non_literal_has_no_choices() -> None:
    """A union is only a choice set when EVERY non-None member is a Literal.

    Falling back to the member itself promoted a plain type into an enum, so
    ``str | None`` emitted ``{"enum": [<class 'str'>]}`` and then rejected a
    real string with "Valid: <class 'str'>".
    """
    assert literal_values(str | None) == ()
    assert Field[str](annotation=str | None).choices == ()
    # Mixed unions are not choice sets either, however literal one side is.
    assert literal_values(_Color | int) == ()


def test_a_float_knob_accepts_a_json_integer() -> None:
    """JSON has one number syntax, so ``1`` is as valid a ``number`` as ``1.0``.

    The schema advertised ``number`` -- which admits an integer -- and the
    check then rejected ``1`` for not being a ``float``.
    """
    field = Field[float](annotation=float)
    assert field.schema() == {"type": "number"}
    assert field.coerce("temperature", 1) == 1


def test_a_tuple_knob_accepts_the_list_json_decodes_to() -> None:
    """Every JSON array decodes to a ``list``, never a ``tuple``.

    The schema advertised ``array`` and the check demanded ``tuple``, so no
    directive could ever satisfy it.
    """
    field = Field[tuple[str, ...]](annotation=tuple)
    assert field.schema() == {"type": "array"}
    assert field.coerce("items", ["a"]) == ["a"]


def test_a_non_literal_union_advertises_no_shape_it_cannot_enforce() -> None:
    """No single JSON type describes ``str | int``, so the schema stays open.

    It previously rendered as ``{"type": "object"}`` and raised
    "expected Union, got list" -- a message naming a typing construct.
    """
    field = Field[object](annotation=str | int)
    assert field.schema() == {}
    assert field.coerce("value", []) == []


def test_object_annotation_leaves_the_schema_type_open() -> None:
    """A JSON body is as legitimately a list or a string as a mapping."""
    field = Field[object](annotation=object, description="Any JSON.")
    assert field.schema() == {"description": "Any JSON."}


def test_schema_extra_merges_shape_the_type_cannot_carry() -> None:
    field = Field[dict[str, str]](
        annotation=dict, schema_extra={"additionalProperties": {"type": "string"}}
    )
    assert field.schema() == {
        "type": "object",
        "additionalProperties": {"type": "string"},
    }


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
