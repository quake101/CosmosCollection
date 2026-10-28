# Getting Started with Cosmos Collection

This guide takes you from a fresh download to a working collection: your location set, your gear added, your images in, and a plan for tonight. It takes about 15 minutes, and almost everything except your location is optional. Skip what you don't need and come back to it later.

**Contents**

1. [Install and open the app](#1-install-and-open-the-app)
2. [Find your way around](#2-find-your-way-around)
3. [Add your location](#3-add-your-location)
4. [Add your equipment](#4-add-your-equipment)
5. [Bring in your images](#5-bring-in-your-images)
6. [Plan a night out](#6-plan-a-night-out)
7. [Optional setup](#7-optional-setup)
8. [Tips and getting help](#8-tips-and-getting-help)

---

## 1. Install and open the app

1. Download the zip for your system from the [Releases](https://github.com/quake101/CosmosCollection/releases) page: `CosmosCollection-Windows.zip`, `CosmosCollection-macOS.zip` or `CosmosCollection-Linux.zip`.
2. Extract it anywhere you like, then run **CosmosCollection**.

On **macOS**, the first launch needs a right-click → **Open**. On **Linux**, you may need one system library first. The [README](../README.md#getting-started) has both steps.

You don't need to reinstall to update. The app checks for new versions when it starts, and can install them for you.

## 2. Find your way around

The main window is a list of deep-sky objects from the built-in catalogs (Messier, NGC, IC, Sharpless, Barnard, Abell and more).

- **Search** by name or designation, such as `M 31`, `NGC 7000` or `Andromeda`. If the catalogs don't have the object, the app offers to look it up on SIMBAD.
- Narrow the list with **Catalog** and **Type**, or use **Filters ▾** to show only objects you have (or haven't) imaged.
- **Double-click an object** to open its **DSO Detail** page: your images of it, when it's visible, and links to the other tools.
- **Right-click an object** to open it in the Visibility Calculator or FOV Simulator, add it to your Target List, or plan a session for it.

Every other tool is on the toolbar along the top:

| Toolbar button | What it's for |
|---|---|
| **Settings** | Your location, plate solving, integrations, theme, backups |
| **Equipment** | Your telescopes, cameras, eyepieces and barlows/reducers |
| **Best DSO Tonight** | The best objects to image from your location tonight |
| **Target List** | Objects you want to image or observe someday |
| **Weather** | A 7-day forecast with an astro score |
| **Session Manager** | Your imaging sessions: planned, in progress and done |
| **NINA Dashboard** | A live view of NINA while you image (shown once NINA is enabled) |
| **Image Gallery** | All your finished images in one place |
| **Collage Builder** | Arrange your images into a poster or collage |
| **Import Wizard** | Bring in an existing library of images and subs |
| **Visibility Calculator** | When an object is up and best placed |
| **FOV Simulator** | Frame a target with your telescope and camera |

## 3. Add your location

The planning tools need to know where you observe from. This is the one step you shouldn't skip.

1. Click **Settings** and open the **Location Manager** tab.
2. Click **Select Location from Map** and click your observing site, or type in your **Latitude** and **Longitude**.
3. Give it a **Name** (such as "Backyard") and check the **Time Zone**.
4. Click **Save Location**.

Your first location becomes the active one automatically. If you image from more than one place (home, a dark site, a star party), add each one and use **Set as Active** to switch between them.

**Have trees or buildings in the way?** Click **Import Horizon...** to load a custom horizon file from NINA or Stellarium. The visibility tools will then treat an object as visible only once it clears your real horizon, not just a flat minimum altitude.

## 4. Add your equipment

Adding your gear lets the app file images under the right telescope and camera, show your real field of view in the FOV Simulator, and keep sessions shot with different rigs apart.

1. Click **Equipment**.
2. On the **Telescopes** tab, click **Add**:
   - **New Telescope** for a regular scope. Enter its **Aperture** and **Focal length** (the focal ratio is worked out for you) and pick the **Mount** type.
   - **Smart Telescope** for a Seestar, Dwarf, Vespera, Celestron Origin and similar. This adds the telescope *and* its built-in camera in one click.
3. On the **Cameras** tab, add your camera with its **Sensor size**, **Resolution** and **Pixel size**.
4. Eyepieces and barlows/reducers are optional, mainly for visual observers.

There's no Save button. Changes are saved as you go.

Don't worry about getting everything perfect now. When you import files that name gear you haven't added, the app asks whether to match them to existing equipment or add it.

## 5. Bring in your images

Pick whichever of these fits you. You can use more than one.

### A. You already have a library of images: use the Import Wizard

This is the fastest way to start if you've been imaging for a while. **Your files are never moved or changed.** The app only reads them and remembers where they are.

1. Click **Import Wizard**.
2. Click **Add Folder...** and pick the folders where you keep your astrophotography. Add as many as you like.
3. Click **Scan**. The app looks through them for imaging sessions (your subs) and finished images. A big library can take a while, and you can click **Stop and Review** at any time.
4. **Match your equipment.** Your files name the telescope and camera they were taken with. Check that each name is matched to the right gear (likely matches are already picked), then click **Next**.
5. **Review what was found.** Everything is grouped by object, with dates, nights, subs, integration time and gear. Uncheck anything you don't want. If some images couldn't be matched to an object, click **Identify by Plate Solving** to work out what they show.
6. Click **Import Selected**.

Sessions go to the **Session Manager**, and finished images go to the **Image Gallery** and each object's DSO Detail page.

Took more pictures since? Open the wizard again and click **Scan Again**. It only picks up what's new.

### B. Adding one finished image

Drag an image file (JPG, PNG, TIFF, FITS or XISF) onto the main window. Or click **Add Image** in the Image Gallery, or **+** on an object's DSO Detail page.

In the **Add Image to DSO** dialog:

- **Attach to DSO** is filled in for you when the app can tell the object from the file's FITS/XISF header or its file name. Check that it's right.
- The app plate solves the image to find **other catalog objects in the frame**. Check any you'd like the image added to as well (a wide field of M 42 might also show M 43 and NGC 1977).
- Fill in **Telescope**, **Camera**, **Integration Time** (such as `2h 30m`), **Date Taken** and any **Notes**, then click **OK**.

If you imported the session the image came from, **Import from Session...** fills in these details for you.

### C. Logging a night's subs: use the Session Manager

After a night of imaging, open **Session Manager** and **drag your sub files or folder onto it**. The app reads the FITS/XISF headers and creates a session, or adds the night to an existing one for the same target. It works out the object, filters, exposures, gear, and whether you shot on an equatorial or alt-az mount.

From a session you can:

- Click **Review Lights Quality...** to grade the subs with graphs and find bad frames (clouds, trailing, poor focus) before you stack.
- Right-click it and choose **Processing** to hand it off to Siril or PixInsight for stacking (set them up first under Settings → Integrations).
- Use **New Planned Session** to plan future nights on the **Calendar** tab, with weather and visibility shown.

<img src="../screenshots/Session-Manager.jpg" width="700" alt="Session Manager">

## 6. Plan a night out

Once your location is set, the planning tools work right away.

**Best DSO Tonight** shows what's well placed from your location tonight, with the Moon's phase and distance from each object. Set a **Min Altitude**, a **Max Magnitude** and the **Type** of object you're after, then click **Calculate Best DSOs Tonight**.

<img src="../screenshots/Best-DSO-Tonight.jpg" width="700" alt="Best DSO Tonight">

A good routine for a night out:

1. Check **Weather** for a clear night. Each day gets an astro score, and you can open the hourly details.
2. Pick a target in **Best DSO Tonight**, or from your **Target List**.
3. Open it in the **Visibility Calculator** to see when it's highest and how long you have before it sets or hits your horizon.
4. Open it in the **FOV Simulator** to frame it with your telescope and camera. You can rotate the frame or plan a mosaic if it doesn't fit.
5. If you use NINA, send the framing straight to NINA's Framing Assistant (see below).

Found an object you'd like to shoot someday? Right-click it anywhere and choose **Add to Target List**.

## 7. Optional setup

All of these are under **Settings**.

**Plate solving** (Integrations tab → Plate Solving). Plate solving works out exactly where an image points, so the app can label the objects in it. The **built-in solver** works out of the box with nothing to install. It uses an online star catalog and usually takes a few seconds. You can also add:
- **ASTAP**, which works offline once installed and doesn't need to know the object in advance.
- **Astrometry.net**, which needs a free API key and can take a few minutes.

**NINA** (Integrations tab → NINA). Turn on **Enable NINA Integration** and enter NINA's API host and port, then click **Test Connection**. You'll get the **NINA Dashboard** for watching a session live, plus right-click options to send targets and mosaic panels to NINA's Framing Assistant or slew to them.

**Siril / PixInsight** (Integrations tab). Point the app at the program and a workspace folder, then click **Test**. Sessions can then be handed off for stacking from the Session Manager.

**Weather.** The forecast works with no setup. If you'd like extra sources, you can add OpenWeather, VisualCrossing or WeatherAPI with your own free API key.

**Look and feel.** The **Theme** tab changes the app's colors. **Application Settings** has 12/24-hour time, weather units, minimize to system tray, and **Offline mode** if you image somewhere without internet.

**Backups** (Backup & Restore tab). Your catalog, sessions and notes live in the app's own database, separate from your image files. Click **Backup Now...** every so often, especially before reinstalling or moving to a new computer.

## 8. Tips and getting help

- **Moved or renamed your image folders?** The app can find moved images and relink them. Use **Check Files...** in a session, or **Relocate Image** on a DSO Detail page.
- **Lots of unsolved images?** The Image Gallery's **Plate Solve Unsolved…** button solves them all in the background, and **Show Annotations** labels the objects in each one.
- **Make a poster.** The **Collage Builder** can start from suggestions like a Messier poster filled with your own images.
- **Hover over things.** Most buttons and settings have a tooltip explaining what they do.

**Found a bug, or have an idea?** Please [open an issue](https://github.com/quake101/CosmosCollection/issues). You don't need to be a developer, and feedback from people using the app is the most useful kind.

Clear skies!
