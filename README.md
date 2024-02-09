# code2fab

## Server
To get descriptions, we would need to send the openscad code to the server like this:

```
curl -i -H "Content-Type: application/json" -X POST -d '{"code":"$fn=32; cube(); sphere();"}' http://128.46.81.40:8880/code2fab
```

It should return a json response containing an llm description after a few minutes, similar to this:

```
{"description":"The object under discussion can be imagined as a combination of two distinct geometric shapes - a perfect sphere and a flat square - each having differing properties. \n\nBegin by picturing a perfect sphere, much akin to a soccer ball or basketball, but devoid of any textual elements like seams or patterns. This sphere represents a three-dimensional round shape, where every point on its surface is equidistant from its center. To grasp its physical manifestation, consider a smooth, continuous surface with no edges or corners, allowing your hand to glide over it without any sudden changes.\n\nAccompanying this sphere is a flat square shape. Visualize a standard board game tile or a drink coaster, which has equal sides and four right-angled corners granting it a distinct flatness. The square and sphere blend together in a way that may be unusual to comprehend. The square overlaps a portion of the sphere, contradicting the sphere's curvature due to its flat construct. It creates an intersection on the sphere's surface, partially concealing it beneath a straight edge as if it were pressed into the sphere. \n\nTo provide a tactile analogy for a blind user, it might feel like holding a ball with a thin, stiff card resting on one side. This card doesn't wrap around the curved surface, instead, it cuts through it, making an intersection. The sphere would be analogous to the ball \u2013 a rounded, full contour, while the flat, angular card simulates the shape of the square."}
```
