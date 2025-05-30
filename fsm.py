from states import State

FSM = {
    State.BACKGROUND: {
        State.UNDEFINED:         State.UNDEFINED,
        State.MANY_TEXT:         State.MANY_TEXT,
        State.FEW_TEXT:          State.FEW_TEXT,
        State.LONG_BLACK_LINE:   State.LONG_BLACK_LINE,
        State.MEDIUM_BLACK_LINE: State.MEDIUM_BLACK_LINE,
        State.COLOR:             State.COLOR,
        State.BACKGROUND:        State.BACKGROUND,
    },
    State.UNDEFINED: {
        State.UNDEFINED:         State.UNDEFINED,
        State.MANY_TEXT:         State.MANY_TEXT,
        State.FEW_TEXT:          State.UNDEFINED,
        State.LONG_BLACK_LINE:   State.UNDEFINED,
        State.MEDIUM_BLACK_LINE: State.MEDIUM_BLACK_LINE,
        State.COLOR:             State.COLOR,
        State.BACKGROUND:        State.BACKGROUND,
        # NOTE: State.BACKGROUND requires more complex logic
    },
    State.MANY_TEXT: {
        State.UNDEFINED:         State.MANY_TEXT,
        State.MANY_TEXT:         State.MANY_TEXT,
        State.FEW_TEXT:          State.MANY_TEXT,
        State.LONG_BLACK_LINE:   State.MANY_TEXT,
        State.MEDIUM_BLACK_LINE: State.MEDIUM_BLACK_LINE, # XXX: ?
        State.COLOR:             State.MANY_TEXT,
        State.BACKGROUND:        State.BACKGROUND,
    },
    State.FEW_TEXT: {
        State.UNDEFINED:         State.UNDEFINED, # XXX: ?
        State.MANY_TEXT:         State.MANY_TEXT,
        State.FEW_TEXT:          State.FEW_TEXT,
        State.LONG_BLACK_LINE:   State.LONG_BLACK_LINE,
        State.MEDIUM_BLACK_LINE: State.MEDIUM_BLACK_LINE,
        State.COLOR:             State.COLOR,
        State.BACKGROUND:        State.BACKGROUND,
    },
    State.LONG_BLACK_LINE: {
        State.UNDEFINED:         State.LONG_BLACK_LINE,
        State.MANY_TEXT:         State.LONG_BLACK_LINE,
        State.FEW_TEXT:          State.LONG_BLACK_LINE,
        State.LONG_BLACK_LINE:   State.LONG_BLACK_LINE,
        State.MEDIUM_BLACK_LINE: State.MEDIUM_BLACK_LINE, # XXX: ?
        State.COLOR:             State.LONG_BLACK_LINE,
        State.BACKGROUND:        State.BACKGROUND,
    },
    State.MEDIUM_BLACK_LINE: {
        State.UNDEFINED:         State.MEDIUM_BLACK_LINE,
        State.MANY_TEXT:         State.MEDIUM_BLACK_LINE,
        State.FEW_TEXT:          State.MEDIUM_BLACK_LINE,
        State.LONG_BLACK_LINE:   State.LONG_BLACK_LINE,
        State.MEDIUM_BLACK_LINE: State.MEDIUM_BLACK_LINE,
        State.COLOR:             State.COLOR, # XXX: ?
        State.BACKGROUND:        State.BACKGROUND,
    },
    State.COLOR: {
        State.UNDEFINED:         State.COLOR,
        State.MANY_TEXT:         State.COLOR, # XXX: ?
        State.FEW_TEXT:          State.COLOR, # XXX: ?
        State.LONG_BLACK_LINE:   State.LONG_BLACK_LINE,
        State.MEDIUM_BLACK_LINE: State.MEDIUM_BLACK_LINE,
        State.COLOR:             State.COLOR,
        State.BACKGROUND:        State.BACKGROUND,
    },
}

def assert_not_forbidden_combo(prev, new):
    assert not (prev == State.UNDEFINED and new == State.LONG_BLACK_LINE)
    assert not (prev == State.MANY_TEXT and new == State.LONG_BLACK_LINE)
    assert not (prev == State.MANY_TEXT and new == State.MEDIUM_BLACK_LINE)
    assert not (prev == State.FEW_TEXT and new == State.LONG_BLACK_LINE)
    assert not (prev == State.FEW_TEXT and new == State.MEDIUM_BLACK_LINE)
    assert not (prev == State.MEDIUM_BLACK_LINE and new == State.MANY_TEXT)
    assert not (prev == State.COLOR and new == State.MANY_TEXT)
