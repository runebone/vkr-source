class State:
    UNDEFINED         = 0
    BACKGROUND        = 1
    FEW_TEXT          = 2
    MANY_TEXT         = 3
    COLOR             = 4
    MEDIUM_BLACK_LINE = 5
    LONG_BLACK_LINE   = 6

StateNames = {
    State.UNDEFINED:         "Undefined",
    State.BACKGROUND:        "Background",
    State.FEW_TEXT:          "Few Text",
    State.MANY_TEXT:         "Many Text",
    State.COLOR:             "Color",
    State.MEDIUM_BLACK_LINE: "Medium Black Line",
    State.LONG_BLACK_LINE:   "Long Black Line",
}

class Class:
    UNDEFINED  = 0
    BACKGROUND = 1
    TEXT       = 2
    TABLE      = 3
    CODE       = 4
    DIAGRAM    = 5
    FIGURE     = 6
    PLOT       = 7
    EQUATION   = 8

ClassNames = {
    Class.UNDEFINED:  "Undefined",
    Class.BACKGROUND: "Background",
    Class.TEXT:       "Text",
    Class.TABLE:      "Table",
    Class.CODE:       "Code",
    Class.DIAGRAM:    "Diagram",
    Class.FIGURE:     "Figure",
    Class.PLOT:       "Plot",
    Class.EQUATION:   "Equation",
}
