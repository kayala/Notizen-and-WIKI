# Eagle

Easily Applicable Graphical Layout Editor

## checklist

[PCB check list](https://grizzlypeak.io/notes/pcb-checklist.html)

## BUS naming

It does not matter if the BUS name is different. Same signal names will still be
connected.

## generating BOM

Use the standard ULP script.

## change value for component in schematic

When Eagle complains that the value is not changing, you have to modify the
component and enable "Value: on".

## resync schematic and board

Error appears: "Board and schematic are not consistent! No forward-/backannotation will be performed!"

Run ERC in the schematic

## footprint

- `Smd Size`: column x hight

component size should be drawn in two layers: tPlace and tDocu.

```
          correct
           view

            │
            ▼
┌─────────────────────────┐
│                         │ ◄──  package
├────┬─┬────┬─┬────┬─┬────┤
└────┘ └────┘ └────┘ └────┘ ◄──  pads
```

## name vs value

- name: R1, C1, U7
- value: 10k, 10uF, BME680

## putting component on opposite layer

Use the mirror command.

## hide the airwire from a particular signal

Just select the signal and go to proprieties (double click). There should be a
"Airwires hidden" option to enable.

## resync schematic and board

Rule: when there is a schematic and board, keep both files open, so they will
stay in sync.

If you just started the board design, it is easier to just redo all the work.

If not, it depends on the error. Inconsistencies can be view in the DRC list.
You may try to solve the problem by exporting NetScript in the schematic and
importing (SRC button) on the board.

## copper filling of vias

Some manufacturers may check the DRC > Supply > Generate thermal for vias, which
causes the thermal for all supply plane vias. Supply plane vias thermals are
undesirable.

## nominal values

- width = 12mil
- drill = 20mil
- clearance = 10mil

## pouring copper

Make sure that Options -> Settings -> Misc: "ratsnest calculates polygons" is
enabled.

1. select polygon on the copper layer
2. draw the polygon
3. rename the polygon to the signal name
4. click ratsnest

Every time you click on ratsnest, it will automatically refill the polygon.
When you don't wish to constantly view the polygon anymore, use the `ripup @ *`
command.

## updating footprint

Eagle stores the symbol, footprint, and schematic information inside the project
file itself. This means that once you insert a part, the part information is
copied out of the library and placed in the .sch file. The link to the library
is lost. In order to update the part in the schematic/board, you need to right
click on the part and select "Replace".

This will not affect >NAME and >VALUE, because of the smash command. Those are
considered a different footprint. For that, you need to right click and "restore
position".

## connector with board cutout

If there is a connector that is positioned "inside the board", you must draw the
cutout with a milling and a dimensions line. Drawing may be "open", i.e. not
closed as in a circle or rectangle, but a simple line.

## positioning names in the board

The best way to position names (or values, or anything else) in the board is by
only enabling the Place and Stop layers.

## custom logo in PCB design

[Importing custom images into
eagle](https://learn.sparkfun.com/tutorials/importing-custom-images-into-eagle/all#method-1-svg-to-polygon)

## layers

- 1 Top: copper layer on top of the board
- 2-15 Route: inner copper layers
- 16 Bottom: copper layer on the bottom
- 17 Pads: Though-hole pads
- 18 Vias: vias
- 19 Unrouted: ratsnest for unconnected components
- 20 Dimension: board outline and design rules to keep copper away
- 21-22 tPlace/bPlace: top and bottom silkscreen and component outline
- 23-24 tOrigins/bOrigins: top and bottom of component origins (the crosshairs)
- 25-26 tNames/bNames: component names
- 27-28 tValues/bValues: component values
- 29-30 tStop/bStop: indicate where solder mask should not be applied
- 31-32 tCream/bCream: solder paste for SMD
- 33-34 tFinish/bFinish: data about special finish, eg. specific pad gold finish
- 35-36 tGlue/bGlue: glue mask, used to secure components in place
- 37-38 tTest/bTest: dedicated test points for ICT equipment
- 39-40 tKeepout/bKeepout: keep components away from specific areas
- 41-43 tRestrict/bRestrict/vRestrict: indicate where copper should be removed
- 44 Drills: holes for pads and vias
- 45 Holes: similar to Drills, but only for holes that don't need to conduct
  electricity
- 46 Milling: milling of holes, inner cutouts and any other kind of contour
- 47 Measures: physical measurements
- 48 Document: supplementary documentation (add thickness, stackup requirements,
  solder mask color, silkscreen color, copper type, copper weight, impedance
  control specification, special finish requirements)
- 49-50 ReferenceLC/ReferenceLS: reference marks for fiducials
- 51-52 tDocu/bDocu: documentation for board reviewer eg. mechanical dimensions
  of components and enclosures

## configuration

Configuration is contained at an `eaglerc` file. In Artix Linux it is found at
`~/.local/share/Eagle/settings/**/eaglerc`.
