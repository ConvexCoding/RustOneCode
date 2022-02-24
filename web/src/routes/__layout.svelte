<script context="module" lang="ts">
  import init from '$lib/asplib'
  export async function load() {
    await init()
    return {}
  }
</script>

<script lang="ts">
  import '../app.css'
</script>

<slot />
