# Spotify API Research: What Can and Cannot Be Automated

**Date:** 2026-03-26
**Purpose:** Determine what Spotify for Artists features can be managed via API vs. manual dashboard

---

## Executive Summary

**There is NO public "Spotify for Artists API."** The official Spotify Web API (developer.spotify.com) is consumer-facing -- it lets you search tracks, control playback, manage playlists, and read catalog metadata. It does NOT provide endpoints for artist profile management, Canvas uploads, Discovery Mode, playlist pitching, or analytics dashboards. All artist management features require the Spotify for Artists web dashboard (artists.spotify.com) or are gated behind private Partner/Provider APIs available only to licensed distributors.

**Given Jeff's prior fine for playlist manipulation, any automation of the Spotify for Artists dashboard carries significant risk.** Spotify explicitly flags "automated processes (e.g. bots or scripts)" as violations and issues EUR 10 per-track fines plus potential account suspension.

---

## 1. What the Official Spotify Web API CAN Do

These are the legitimate, documented endpoints at developer.spotify.com:

| Feature | Endpoint | Notes |
|---------|----------|-------|
| Search tracks/artists/albums | `/v1/search` | Public catalog search |
| Get artist info | `/v1/artists/{id}` | Name, genres, followers, images, popularity (popularity removed Feb 2026) |
| Get artist's top tracks | Removed Feb 2026 | Was `/v1/artists/{id}/top-tracks` |
| Get artist's albums | `/v1/artists/{id}/albums` | List of albums |
| Get related artists | `/v1/artists/{id}/related-artists` | Similar artists |
| Control playback | `/v1/me/player/*` | Play, pause, skip, queue |
| Manage user playlists | `/v1/playlists/*` | Create, add/remove tracks |
| Get user's top items | `/v1/me/top/{type}` | User's top artists/tracks |
| Get recommendations | `/v1/recommendations` | Based on seed tracks/artists/genres |
| Follow/unfollow artists | `/v1/me/following` | As a listener, not artist management |

### February 2026 API Lockdown

Spotify removed 15 endpoints and restricted access significantly:
- **Removed:** Artist top tracks, browse categories, album metadata, new releases
- **Removed fields:** Track/album popularity, user country, subscription tier
- **Dev Mode limits:** Premium account required, 1 client ID per developer, 5 authorized users max
- **Extended access:** Requires registered business + 250K monthly active users
- **Rate limits:** Dropped from ~180 to ~100 requests/minute

### Available MCP Spotify Tools

The Spotify MCP tools you have available are consumer-level:
- `auth-spotify`, `search-spotify`, `play-track`
- `get-user-playlists`, `create-playlist`, `add-tracks-to-playlist`
- `get-recommendations`, `get-top-tracks`
- `get-current-playback`, `next-track`, `previous-track`, `pause-playback`

**None of these touch artist management.** They are all listener/consumer functions.

---

## 2. What CANNOT Be Done via API (Requires Dashboard / Manual)

### Artist Profile Management
- **Update bio, profile photo, header image:** Dashboard only (artists.spotify.com)
- **Artist Pick (pinned item):** Dashboard only
- **Social links:** Dashboard only
- **No API exists** for any profile management

### Canvas Video Uploads
- **Upload Canvas videos:** Dashboard only
- Canvas must be uploaded per-track through the Spotify for Artists web UI
- Processing takes ~10+ minutes per upload
- **Unofficial fetch-only:** GitHub projects can *download/fetch* Canvas video URLs using reverse-engineered Protobuf APIs, but these are read-only and violate ToS
- **No upload API exists** (official or unofficial)

### Discovery Mode
- **Enable/disable Discovery Mode:** Dashboard only at artists.spotify.com > Campaigns > Discovery Mode
- Requires 25,000+ monthly listeners to be eligible
- 30% commission on royalties from Discovery Mode streams
- **No API exists** -- must be set up manually per campaign

### Editorial Playlist Pitching
- **Pitch tracks to editors:** Dashboard only (Music > Upcoming > Pitch a Song)
- Can only pitch ONE unreleased track at a time
- Must submit 7+ days before release date
- **No API exists** -- this is an editorial/human process
- Some distributors (DistroKid, TuneCore, CD Baby) offer pitching services as part of their platforms

### Detailed Analytics / Stream Counts
- **Real-time streams, listener demographics, playlist adds, save rates:** Dashboard only
- The public Web API never exposed per-track stream counts or detailed analytics
- Popularity scores (a rough proxy) were removed in Feb 2026

---

## 3. Unofficial / Reverse-Engineered Options

### PouleR/spotify-artists-api (PHP)
- **GitHub:** https://github.com/PouleR/spotify-artists-api
- Unofficial PHP wrapper that reverse-engineers the Spotify for Artists internal API
- Can fetch: upcoming releases, real-time statistics
- Requires login credentials (SpotifyLogin class to obtain access token)
- **Risk:** Uses undocumented internal APIs; could break at any time; account could be flagged

### SpotAPI (Python)
- **GitHub:** https://github.com/Aran404/SpotAPI
- Interacts with private Spotify APIs, emulating browser requests
- Primarily consumer-side, not artist management

### Spotipy (Python - Official API Wrapper)
- **GitHub:** https://github.com/spotipy-dev/spotipy
- Well-maintained wrapper for the OFFICIAL Web API only
- Cannot do anything the official API doesn't support
- Good for: search, playback, playlist management, catalog data

### spotify-web-api-node (Node.js)
- **GitHub:** https://github.com/thelinmichael/spotify-web-api-node
- Same as Spotipy but for Node -- official API wrapper only

### Browser Automation (Selenium/Playwright)
- People have used Selenium and Playwright to scrape Spotify playlists and catalog data
- **Could theoretically automate the Spotify for Artists dashboard** (login, upload Canvas, enable Discovery Mode, etc.)
- **EXTREMELY RISKY** -- see section 4 below

---

## 4. Risk Assessment

### What Could Get Jeff Flagged/Fined

| Action | Risk Level | Consequence |
|--------|-----------|-------------|
| Using official Web API for search/playback | **SAFE** | Normal usage |
| Scraping public catalog data via API | **LOW** | Rate limiting at worst |
| Using unofficial artist API (PouleR) for analytics | **MEDIUM** | Account could be flagged; internal APIs change without notice |
| Browser automation of Spotify for Artists dashboard | **HIGH** | Account suspension; Spotify explicitly flags "automated processes" |
| Automating Discovery Mode enablement | **HIGH** | Could be seen as manipulation |
| Automating Canvas uploads via browser | **MEDIUM-HIGH** | Less likely to be seen as manipulation but still violates ToS |
| Automating playlist pitches | **HIGH** | Editorial process -- gaming this could result in blacklisting |
| Any form of artificial streaming | **CRITICAL** | EUR 10/track fine, royalty withholding, content removal, account termination |

### Spotify's Current Stance on Automation (Feb 2026)

Spotify explicitly stated: "Advances in automation and AI have fundamentally altered the usage patterns and risk profile of developer access." They are actively cracking down. Key policies:
- EUR 10 per-track fine for artificial streaming detection
- Zero-tolerance policy since 2024
- "Any instance of attempting to manipulate Spotify by using automated processes (e.g. bots or scripts)" is flagged
- Account suspension/termination for repeated violations
- Spotify's new "Artist Profile Protection" beta (March 2026) further monitors for suspicious activity

---

## 5. Distributor API Options

Spotify doesn't offer a public distribution API. Everything goes through licensed distributors:

| Distributor | API Available? | Notes |
|-------------|---------------|-------|
| DistroKid | No public API | Has internal enterprise API for their platform |
| TuneCore | No public API | Spotify Preferred Provider |
| CD Baby | No public API | Some features accessible via their dashboard |
| FUGA | Enterprise API | Available to labels, not individual artists |
| Revelator | REST API | White-label distribution API with sandbox |
| Amuse | No public API | Offers pitching services |

None of these give individual artists programmatic access to Spotify for Artists features. They handle distribution (getting music onto Spotify) but not artist dashboard management.

---

## 6. Recommended Approach for Jeff

### DO (Safe)
1. **Use the official Spotify Web API** via the MCP tools for searching, discovering music, managing personal playlists, and playback control
2. **Use Spotipy** if you need to build any catalog-data features into StemScribe (e.g., linking to Spotify tracks)
3. **Manually manage your Spotify for Artists dashboard** -- bio, Canvas, Discovery Mode, playlist pitches
4. **Use DistroKid's built-in tools** for playlist pitching if available through your distributor
5. **Pull public data** (follower counts, genres, album listings) through the official API for display purposes

### DO NOT (Given Prior Fine History)
1. **Do NOT automate the Spotify for Artists dashboard** with Selenium/Playwright -- this is the fastest path to another fine or account suspension
2. **Do NOT use unofficial artist APIs** (PouleR, SpotAPI) for anything that modifies your account
3. **Do NOT attempt to programmatically enable Discovery Mode** or submit playlist pitches
4. **Do NOT use any third-party service that "guarantees" streams or playlist placements
5. **Do NOT scrape Spotify for Artists analytics** -- if you need this data, screenshot it or export CSVs manually

### CONSIDER (Medium Risk, Potentially Worth It)
1. **Read-only unofficial analytics** via PouleR's library -- lower risk since you're only reading your own data, but still uses undocumented APIs
2. **Chartmetric or similar third-party analytics tools** -- these are legitimate businesses that aggregate public data and have their own API agreements with Spotify
3. **Artist.tools or similar services** -- legitimate analytics platforms for tracking your streams

---

## 7. Bottom Line

Spotify has been aggressively locking down API access throughout 2025-2026. The trend is toward LESS openness, not more. There is no legitimate way to automate artist profile management, Canvas uploads, Discovery Mode, or playlist pitching. The only safe API usage is the official Web API for consumer-level features (search, playback, playlists).

For Jeff's situation specifically -- with a prior fine for playlist manipulation -- the safest path is to do everything manually through the Spotify for Artists dashboard and focus automation efforts on other platforms where APIs are more permissive.

---

## Sources

- [Spotify Web API Documentation](https://developer.spotify.com/documentation/web-api)
- [Spotify Web API Reference - Get Artist](https://developer.spotify.com/documentation/web-api/reference/get-an-artist)
- [Spotify for Artists API Community Thread](https://community.spotify.com/t5/Spotify-for-Developers/Spotify-for-artists-API/td-p/5067109)
- [Spotify API Lock-Down (Feb 2026)](https://medium.com/@apollinereymond/spotifys-api-lock-down-the-end-of-open-data-for-the-music-business-0a9bf07dba27)
- [February 2026 Web API Changelog](https://developer.spotify.com/documentation/web-api/references/changes/february-2026)
- [February 2026 Migration Guide](https://developer.spotify.com/documentation/web-api/tutorials/february-2026-migration-guide)
- [State of Spotify Web API Report 2025](https://spotify.leemartin.com/)
- [Extended Access Criteria Update (Apr 2025)](https://developer.spotify.com/blog/2025-04-15-updating-the-criteria-for-web-api-extended-access)
- [Update on Developer Access (Feb 2026)](https://developer.spotify.com/blog/2026-02-06-update-on-developer-access-and-platform-security)
- [PouleR/spotify-artists-api (GitHub)](https://github.com/PouleR/spotify-artists-api)
- [Spotipy Python Library](https://github.com/spotipy-dev/spotipy)
- [SpotAPI Unofficial Python Library](https://github.com/Aran404/SpotAPI)
- [spotify-web-api-node (GitHub)](https://github.com/thelinmichael/spotify-web-api-node)
- [Using Discovery Mode](https://support.spotify.com/us/artists/article/using-discovery-mode-in-spotify-for-artists/)
- [Discovery Mode via DistroKid](https://support.distrokid.com/hc/en-us/articles/17653472330771-Accessing-Spotify-Discovery-Mode)
- [Pitching to Playlist Editors](https://support.spotify.com/us/artists/article/pitching-music-to-playlist-editors/)
- [Spotify Artificial Streaming Policy](https://support.spotify.com/us/artists/article/third-party-services-that-guarantee-streams/)
- [Artificial Streaming Penalty (FUGA)](https://support.fuga.com/hc/en-us/articles/36690008503700-Understanding-Spotify-s-Artificial-Streaming-Penalty-and-FUGA-s-Enforcement-Policy)
- [Avoiding Spotify Penalty Fines (EmuBands)](https://www.emubands.com/faqs/how-to-avoid-spotify-penalty-fines/)
- [Canvas Content Policy](https://support.spotify.com/us/artists/article/canvas-content-policy/)
- [Spotify Platform Rules](https://support.spotify.com/article/platform-rules/)
- [Spotify Artist Profile Protection (March 2026)](https://www.digitalmusicnews.com/2026/03/24/spotify-artist-profile-protection/)
- [Spotify on AI Risks and Developer Restrictions](https://musically.com/2026/02/12/spotify-says-ai-risks-are-reason-for-new-developer-restrictions/)
