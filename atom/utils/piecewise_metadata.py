# SPDX-License-Identifier: MIT
"""Declarative metadata views for the common AF piecewise runner.

Models declare fields, dictionary selections, fresh caches and result writes.
This module owns reconstruction and publication; it has no model dependencies.
"""

from copy import copy
from dataclasses import is_dataclass, replace

from atom.utils.piecewise_af import PiecewiseMetadata


def _field(obj, name):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def _set_field(obj, name, value):
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def _resolve(value):
    """Compile a dotted field path once; callables handle model host decisions."""
    if callable(value):
        return value
    if not isinstance(value, str) or not value or any(not p for p in value.split(".")):
        raise TypeError("Metadata selectors must be nonempty dotted paths or callables")
    parts = value.split(".")

    def read(scope):
        obj = scope
        for part in parts:
            if obj is None:
                return None
            obj = _field(obj, part)
        return obj

    return read


def _names(values, label):
    if isinstance(values, str) or not all(isinstance(v, str) for v in values):
        raise TypeError(f"{label} must be a sequence of field names")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate fields in {label}")
    return tuple(values)


def _check_options(options, allowed, label):
    unknown = options.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown {label} options: {sorted(unknown)}")


def _selection(options):
    _check_options(options, {"keys", "suffix", "when"}, "selection")
    if ("keys" in options) == ("suffix" in options):
        raise ValueError("A metadata selection requires exactly one of keys or suffix")
    when = _resolve(options["when"]) if "when" in options else lambda scope: True
    if "keys" in options:
        if isinstance(options["keys"], str):
            raise TypeError("Selection keys must be a sequence of selectors")
        keys = tuple(_resolve(key) for key in options["keys"])

        def select(values, scope):
            return (
                {key: values[key] for key in (read(scope) for read in keys)}
                if when(scope)
                else {}
            )

    else:
        suffix = _resolve(options["suffix"])

        def select(values, scope):
            if not when(scope):
                return {}
            ending = suffix(scope)
            return {key: values[key] for key in sorted(values) if key.endswith(ending)}

    return select


def _fresh(value):
    if type(value) is dict:
        return {key: _fresh(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return type(value)(_fresh(item) for item in value)
    if type(value) not in (str, int, float, bool, type(None)):
        raise TypeError(
            "Metadata reset values must contain only containers and scalars"
        )
    return value


def _clone(template, updates):
    if is_dataclass(template) and not isinstance(template, type):
        return replace(template, **updates)
    result = copy(template)
    for name, value in updates.items():
        _set_field(result, name, value)
    return result


class PiecewiseMetadataSchema:
    """Compile a ``piecewise_core(metadata={...})`` declaration.

    Paths start at bound core arguments or ``layer``. Ordered ``bindings`` add
    reusable selectors to that scope. ``objects`` declares capture views in
    dependency order: ``inputs`` names fields, ``select`` filters dictionary
    fields, and ``reset`` gives fresh values for derived state on EACH execution.
    Undeclared fields remain on a shallow copy of the live source object.

    ``bind`` attaches earlier capture views, either by name or as
    ``{"object": name, "key": selector}`` for a one-entry dict (empty for None).
    ``template`` clones an earlier view while reading inputs from ``source``;
    this preserves explicitly declared aliases. A None source yields None.

    ``arguments`` lists views passed to the core. ``publish`` maps their field
    paths to ``update``, ``replace``, ``if_not_none``, or ``{"keys": (...)}``.
    Keyed writes skip absent results and leave other live entries intact.
    ``key``, ``when`` and ``memo`` have the same role as in PiecewiseMetadata.
    Schemas never evaluate strings as Python code or walk undeclared objects.
    """

    def __init__(
        self,
        *,
        objects,
        arguments,
        bindings=None,
        key=(),
        when=None,
        memo=None,
        copy_each_call=(),
        publish=None,
    ):
        self.bindings = tuple(
            (name, _resolve(v)) for name, v in (bindings or {}).items()
        )
        if isinstance(key, str):
            raise TypeError("Metadata key must be a sequence of selectors")
        self.key = tuple(_resolve(v) for v in key)
        self.when = _resolve(when) if when is not None else lambda scope: True
        self.memo = _resolve(memo) if memo is not None else lambda scope: None
        self.copy_each_call = copy_each_call
        self.arguments = _names(arguments, "arguments")
        if set(self.arguments) - objects.keys():
            raise ValueError("Metadata arguments must name declared objects")
        self.objects = []
        known = set()
        for name, obj in objects.items():
            _check_options(
                obj, {"source", "inputs", "select", "reset", "bind", "template"}, name
            )
            inputs = _names(obj.get("inputs", ()), f"{name}.inputs")
            selected = tuple(
                (k, _selection(v)) for k, v in obj.get("select", {}).items()
            )
            reset = _fresh(obj.get("reset", {}))
            links = []
            for field, link in obj.get("bind", {}).items():
                if isinstance(link, str):
                    target, get_key = link, None
                else:
                    _check_options(link, {"object", "key"}, "bind")
                    target, get_key = link["object"], _resolve(link["key"])
                if target not in known:
                    raise ValueError(
                        f"{name}.{field} must bind an earlier metadata object"
                    )
                links.append((field, target, get_key))
            fields = (
                *inputs,
                *(k for k, _ in selected),
                *reset,
                *(k for k, _, _ in links),
            )
            _names(fields, f"{name} fields")
            template = obj.get("template")
            if template is not None and template not in known:
                raise ValueError(
                    f"{name}.template must name an earlier metadata object"
                )
            self.objects.append(
                (
                    name,
                    _resolve(obj.get("source", name)),
                    inputs,
                    selected,
                    reset,
                    links,
                    template,
                )
            )
            known.add(name)
        self.publications = []
        for path, rule in (publish or {}).items():
            root, sep, field = path.partition(".")
            if not sep or root not in self.arguments or not field:
                raise ValueError(
                    "Publication paths must name a returned object's field"
                )
            parent, _, leaf = field.rpartition(".")
            read_parent = _resolve(parent) if parent else None
            keys = None
            if isinstance(rule, dict):
                _check_options(rule, {"keys"}, "publication")
                if isinstance(rule["keys"], str):
                    raise TypeError("Publication keys must be a sequence of selectors")
                keys = tuple(_resolve(v) for v in rule["keys"])
                mode = "keys"
            else:
                mode = rule
                if mode not in ("update", "replace", "if_not_none"):
                    raise ValueError(f"Unknown metadata publication mode: {mode}")
            self.publications.append((root, read_parent, leaf, mode, keys))

    def __call__(self, layer, arguments):
        scope = {"layer": layer, **arguments}
        for name, read in self.bindings:
            scope[name] = read(scope)
        if not self.when(scope):
            return None
        inputs, templates, links = {}, {}, {}
        for name, read, fields, selections, _, bindings, _ in self.objects:
            source = templates[name] = read(scope)
            inputs[name] = (
                None
                if source is None
                else {
                    **{field: _field(source, field) for field in fields},
                    **{
                        field: select(_field(source, field), scope)
                        for field, select in selections
                    },
                }
            )
            links[name] = (
                tuple(
                    (
                        field,
                        target,
                        get_key(scope) if get_key else None,
                        get_key is not None,
                    )
                    for field, target, get_key in bindings
                )
                if source is not None
                else ()
            )
        publications = tuple(
            (
                root,
                parent,
                leaf,
                mode,
                tuple(read(scope) for read in keys) if keys is not None else (),
            )
            for root, parent, leaf, mode, keys in self.publications
        )

        def rebuild(values):
            built = {}
            for name, _, _, _, reset, _, template in self.objects:
                if templates[name] is None:
                    built[name] = None
                    continue
                updates = {**values[name], **_fresh(reset)}
                for field, target, item_key, as_dict in links[name]:
                    value = built[target]
                    updates[field] = (
                        ({} if value is None else {item_key: value})
                        if as_dict
                        else value
                    )
                built[name] = _clone(
                    built[template] if template else templates[name], updates
                )
            return {name: built[name] for name in self.arguments}

        def publish(results):
            for root, parent, leaf, mode, keys in publications:
                source, target = results[root], templates[root]
                if source is None or target is None:
                    continue
                if parent is not None:
                    source, target = parent(source), parent(target)
                value = _field(source, leaf)
                if mode == "keys":
                    destination = _field(target, leaf)
                    for item_key in keys:
                        if item_key in value:
                            destination[item_key] = value[item_key]
                elif mode == "update":
                    _field(target, leaf).update(value)
                elif mode == "replace" or value is not None:
                    _set_field(target, leaf, value)

        return PiecewiseMetadata(
            key=(
                tuple(read(scope) for read in self.key),
                tuple(type(value) for value in templates.values()),
                tuple(links.items()),
                publications,
            ),
            inputs=inputs,
            rebuild=rebuild,
            publish=publish if publications else None,
            memo=self.memo(scope),
            copy_each_call=self.copy_each_call,
        )
